"""Evaluate reward configs over synthetic or recorded trajectories, offline.

Mirrors the reward computation in RewardShapingWrapper.step exactly — same
context builder, same raw features, same term-by-term evaluation order and
weight/enabled semantics — without creating a gym env. A proposed task config
can therefore be scored against hand-built or recorded trajectories in
milliseconds, with a per-term breakdown.

Used by the Tier-3 semantic tests (does a "left then right" proposal actually
score alternating motion above one-sided motion?) and designed for reuse in
the revise loop: score a proposal against observed rollout trajectories to
diagnose why training produced the wrong behavior.

A trajectory is a list of step dicts:
    {"obs": [...], "action": <scalar or list>, "native_reward": <float, default 0>}
"""
from __future__ import annotations

from typing import Any

from train_backend_reward_tuning.reward_shaping import (
    enrich_reward_context_with_task_config,
    evaluate_reward_expression,
    normalize_reward_terms,
    reward_expression_context,
)
from train_backend_reward_tuning.reward_templates import raw_reward_terms_for_env


def evaluate_task_config_on_trajectory(
    env_name: str,
    reward_terms: list[dict[str, Any]],
    task_config: dict[str, Any] | None,
    trajectory: list[dict[str, Any]],
    *,
    step_duration_sec: float = 0.02,
    include_per_step: bool = False,
) -> dict[str, Any]:
    terms = normalize_reward_terms(env_name, reward_terms or [])
    per_term_totals: dict[str, float] = {term["key"]: 0.0 for term in terms}
    per_step: list[dict[str, Any]] = []
    total = 0.0
    prev_obs = None
    prev_action = None

    for index, point in enumerate(trajectory, start=1):
        obs = point.get("obs")
        action = point.get("action", 0.0)
        native = float(point.get("native_reward", 0.0))
        raw_terms = raw_reward_terms_for_env(
            env_name=env_name,
            obs=obs,
            action=action,
            previous_action=prev_action,
            native_reward=native,
            info={},
        )
        context = reward_expression_context(
            obs=obs,
            action=action,
            previous_obs=prev_obs,
            previous_action=prev_action,
            native_reward=native,
            raw_terms=raw_terms,
            env_name=env_name,
            step=index,
            time_sec=index * step_duration_sec,
        )
        context = enrich_reward_context_with_task_config(context, task_config)

        step_total = 0.0
        breakdown: dict[str, float] = {}
        for term in terms:
            expression = str(term.get("expression") or "").strip()
            if not expression:
                continue
            try:
                raw_value = evaluate_reward_expression(expression, context)
            except ValueError:
                raw_value = 0.0
            context[term["key"]] = float(raw_value)
            contribution = float(term["weight"]) * float(raw_value) if term["enabled"] else 0.0
            breakdown[term["key"]] = contribution
            per_term_totals[term["key"]] += contribution
            step_total += contribution
        total += step_total
        if include_per_step:
            per_step.append({
                "step": index,
                "time_sec": index * step_duration_sec,
                "total": step_total,
                "breakdown": breakdown,
            })
        prev_obs = obs
        prev_action = action

    result: dict[str, Any] = {
        "total": float(total),
        "mean_per_step": float(total / len(trajectory)) if trajectory else 0.0,
        "per_term_totals": {key: float(value) for key, value in per_term_totals.items()},
        "steps": len(trajectory),
    }
    if include_per_step:
        result["per_step"] = per_step
    return result


def evaluate_proposal_on_trajectory(
    env_name: str,
    proposal: dict[str, Any],
    trajectory: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper for a full proposal dict (as returned by propose_task_config)."""
    task_config = {
        "task_params": proposal.get("task_params") or [],
        "derived_signals": proposal.get("derived_signals") or [],
    }
    return evaluate_task_config_on_trajectory(
        env_name,
        proposal.get("reward_terms") or [],
        task_config,
        trajectory,
        **kwargs,
    )
