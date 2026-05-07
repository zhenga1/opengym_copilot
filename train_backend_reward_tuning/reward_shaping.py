from __future__ import annotations

import ast
import math
import re
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from train_backend_reward_tuning.reward_templates import (
    raw_reward_terms_for_env,
    reward_template_for_env,
)


def infer_episode_outcome(info: dict[str, Any], terminated: bool, truncated: bool) -> tuple[str, str]:
    is_success = info.get("is_success")
    if is_success is True:
        return "success", "environment reported success"
    if is_success is False:
        return "failure", "environment reported failure"
    if truncated and not terminated:
        return "success", "time limit reached"
    if terminated:
        return "failure", "environment terminated"
    if truncated:
        return "success", "episode truncated"
    return "unknown", "outcome unavailable"


def reward_monitor_keys(env_name: str) -> tuple[str, ...]:
    keys = [f"reward_{term['key']}" for term in reward_template_for_env(env_name)]
    keys.append("reward_total")
    return tuple(keys)


_REWARD_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "clip": lambda value, lo, hi: float(np.clip(value, lo, hi)),
    "sqrt": lambda value: float(np.sqrt(value)),
    "square": lambda value: float(np.square(value)),
    "exp": lambda value: float(np.exp(value)),
    "log": lambda value: float(np.log(value)),
    "sin": lambda value: float(np.sin(value)),
    "cos": lambda value: float(np.cos(value)),
    "tanh": lambda value: float(np.tanh(value)),
    "sign": lambda value: float(np.sign(value)),
}
_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)
_COMPILED_EXPRESSION_CACHE: dict[tuple[str, tuple[str, ...]], Any] = {}


def _flat_values(value, *, default_length: int = 1) -> np.ndarray:
    if value is None:
        return np.zeros(default_length, dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def reward_expression_context(
    obs,
    action,
    previous_action,
    native_reward: float,
    raw_terms: dict[str, float] | None = None,
) -> dict[str, float]:
    context: dict[str, float] = {
        "native": float(native_reward),
    }
    for key, value in (raw_terms or {}).items():
        if _REWARD_KEY_PATTERN.match(key):
            context[key] = float(value)

    flat_obs = _flat_values(obs)
    flat_action = _flat_values(action)
    flat_prev_action = _flat_values(previous_action, default_length=flat_action.size or 1)

    for index, value in enumerate(flat_obs):
        context[f"obs_{index}"] = float(value)
    for index, value in enumerate(flat_action):
        context[f"action_{index}"] = float(value)
    for index, value in enumerate(flat_prev_action):
        context[f"prev_action_{index}"] = float(value)

    return context


def _compile_reward_expression(expression: str, allowed_names: set[str]) -> Any:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid reward expression syntax: {exc.msg}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported expression construct: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
                raise ValueError("Only simple math helper calls are allowed in reward expressions.")
        if isinstance(node, ast.Name):
            if node.id not in allowed_names and node.id not in _ALLOWED_FUNCTIONS:
                raise ValueError(f"Unknown reward expression variable: {node.id}")

    return compile(tree, "<reward-expression>", "eval")


def evaluate_reward_expression(expression: str, context: dict[str, float]) -> float:
    cache_key = (expression, tuple(sorted(context)))
    code = _COMPILED_EXPRESSION_CACHE.get(cache_key)
    if code is None:
        code = _compile_reward_expression(expression, set(context))
        _COMPILED_EXPRESSION_CACHE[cache_key] = code
    scope = {**_ALLOWED_FUNCTIONS, **context}
    return float(eval(code, {"__builtins__": {}}, scope))


def reward_expression_variable_names(
    obs_size: int,
    action_size: int,
    *,
    include_previous_action: bool = True,
    raw_term_keys: list[str] | None = None,
) -> list[str]:
    variable_names = ["native"]
    variable_names.extend([f"obs_{index}" for index in range(max(0, obs_size))])
    variable_names.extend([f"action_{index}" for index in range(max(1, action_size))])
    if include_previous_action:
        variable_names.extend([f"prev_action_{index}" for index in range(max(1, action_size))])
    for key in raw_term_keys or []:
        if key != "native":
            variable_names.append(key)
    return variable_names


def validate_reward_terms(
    env_name: str,
    terms: list[dict[str, Any]] | None,
    *,
    available_variable_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    normalized = normalize_reward_terms(env_name, terms)
    seen_keys: set[str] = set()
    allowed_variable_names = set(available_variable_names or [])

    for term in normalized:
        key = term["key"]
        if not _REWARD_KEY_PATTERN.match(key):
            raise ValueError(
                f"Invalid reward key '{key}'. Use letters, numbers, and underscores, and start with a letter or underscore."
            )
        if key in seen_keys:
            raise ValueError(f"Duplicate reward key '{key}'. Reward term keys must be unique.")
        seen_keys.add(key)

        expression = str(term.get("expression") or "").strip()
        if expression:
            evaluate_reward_expression(expression, {name: 0.0 for name in allowed_variable_names})

    return normalized


def normalize_reward_terms(env_name: str, terms: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    template = {term["key"]: term for term in reward_template_for_env(env_name)}
    incoming = {term.get("key"): term for term in (terms or []) if term.get("key")}
    normalized: list[dict[str, Any]] = []

    for key, base_term in template.items():
        override = incoming.get(key, {})
        normalized.append(
            {
                **base_term,
                "weight": float(override.get("weight", base_term["weight"])),
                "enabled": bool(override.get("enabled", base_term["enabled"])),
                "expression": str(override.get("expression", base_term.get("expression", "")) or "").strip(),
                "is_custom": False,
            }
        )

    for key, override in incoming.items():
        if key in template:
            continue
        normalized.append(
            {
                "key": key,
                "label": override.get("label", key.replace("_", " ").title()),
                "description": override.get("description", "Custom reward term."),
                "weight": float(override.get("weight", 0.0)),
                "enabled": bool(override.get("enabled", True)),
                "expression": str(override.get("expression", "") or "").strip(),
                "is_custom": True,
            }
        )

    return normalized


class RewardShapingWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        env_name: str,
        config_provider: Callable[[], list[dict[str, Any]]],
    ) -> None:
        super().__init__(env)
        self.env_name = env_name
        self.config_provider = config_provider
        self._previous_action = None
        self._reset_episode_sums()

    def _reset_episode_sums(self) -> None:
        self._episode_term_sums = {
            term["key"]: 0.0 for term in reward_template_for_env(self.env_name)
        }
        self._episode_reward_history: list[dict[str, Any]] = []

    def reset(self, **kwargs):
        self._previous_action = None
        self._reset_episode_sums()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, native_reward, terminated, truncated, info = self.env.step(action)
        raw_terms = raw_reward_terms_for_env(
            env_name=self.env_name,
            obs=obs,
            action=action,
            previous_action=self._previous_action,
            native_reward=native_reward,
            info=info,
        )
        terms = normalize_reward_terms(self.env_name, self.config_provider())
        expression_context = reward_expression_context(
            obs=obs,
            action=action,
            previous_action=self._previous_action,
            native_reward=native_reward,
            raw_terms=raw_terms,
        )

        for term in terms:
            expression = str(term.get("expression") or "").strip()
            if not expression:
                continue
            try:
                raw_terms[term["key"]] = evaluate_reward_expression(expression, expression_context)
                expression_context[term["key"]] = float(raw_terms[term["key"]])
            except ValueError:
                raw_terms[term["key"]] = 0.0
                expression_context[term["key"]] = 0.0

        for term in terms:
            self._episode_term_sums.setdefault(term["key"], 0.0)

        reward_breakdown: dict[str, float] = {}
        total_reward = 0.0

        for term in terms:
            raw_value = float(raw_terms.get(term["key"], 0.0))
            contribution = float(term["weight"]) * raw_value if term["enabled"] else 0.0
            reward_breakdown[term["key"]] = contribution
            self._episode_term_sums[term["key"]] += contribution
            total_reward += contribution

        info = dict(info)
        info["reward_breakdown"] = {"total": float(total_reward), **reward_breakdown}
        info["reward_raw_terms"] = {key: float(value) for key, value in raw_terms.items()}
        info["reward_weights"] = {
            term["key"]: float(term["weight"]) if term["enabled"] else 0.0 for term in terms
        }
        self._episode_reward_history.append(
            {
                "step": len(self._episode_reward_history) + 1,
                "reward": float(total_reward),
                "reward_breakdown": {"total": float(total_reward), **reward_breakdown},
                "reward_raw_terms": {key: float(value) for key, value in raw_terms.items()},
            }
        )

        if terminated or truncated:
            episode_total = float(sum(self._episode_term_sums.values()))
            episode_outcome, outcome_reason = infer_episode_outcome(info, terminated, truncated)
            info["reward_total"] = episode_total
            for key, value in self._episode_term_sums.items():
                info[f"reward_{key}"] = float(value)
            info["reward_breakdown_episode"] = {
                "total": episode_total,
                **{key: float(value) for key, value in self._episode_term_sums.items()},
            }
            info["reward_history_episode"] = list(self._episode_reward_history)
            info["episode_outcome"] = episode_outcome
            info["episode_outcome_reason"] = outcome_reason
            info["episode_terminal_timestep"] = len(self._episode_reward_history)
            info["episode_terminated"] = bool(terminated)
            info["episode_truncated"] = bool(truncated)

        self._previous_action = action
        return obs, float(total_reward), terminated, truncated, info
