from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pooled_scale(a: list[float], b: list[float]) -> float:
    values = [abs(_safe_float(value)) for value in [*a, *b]]
    return max(mean(values), 1e-6) if values else 1e-6


def _episode_total_for_term(episode: dict[str, Any], term_key: str) -> float:
    breakdown = episode.get("reward_breakdown", {}) or {}
    if term_key in breakdown:
        return _safe_float(breakdown.get(term_key))

    total = 0.0
    for step in episode.get("reward_history", []) or []:
        total += _safe_float((step.get("reward_breakdown", {}) or {}).get(term_key))
    return total


def _terminal_window_total(episode: dict[str, Any], term_key: str, window_size: int = 8) -> float:
    history = episode.get("reward_history", []) or []
    if not history:
        return 0.0
    terminal_slice = history[-window_size:]
    return sum(_safe_float((step.get("reward_breakdown", {}) or {}).get(term_key)) for step in terminal_slice)


def _term_series(episode: dict[str, Any], term_key: str) -> list[float]:
    history = episode.get("reward_history", []) or []
    return [_safe_float((step.get("reward_breakdown", {}) or {}).get(term_key)) for step in history]


def _window_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _build_single_outcome_terminal_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    term_keys = sorted({
        key
        for episode in episodes
        for key in (episode.get("reward_breakdown", {}) or {}).keys()
        if key != "total"
    })
    terminal_timestep_values = [
        _safe_float(episode.get("episode_terminal_timestep", len(episode.get("reward_history", []) or [])))
        for episode in episodes
    ]
    per_term = {}
    for term_key in term_keys:
        terminal_means = []
        baseline_means = []
        worsening = []
        for episode in episodes:
            series = _term_series(episode, term_key)
            if not series:
                continue
            split_index = max(1, len(series) - 8)
            baseline = series[:split_index]
            terminal = series[split_index:]
            baseline_mean = _window_mean(baseline)
            terminal_mean = _window_mean(terminal)
            baseline_means.append(baseline_mean)
            terminal_means.append(terminal_mean)
            worsening.append(terminal_mean - baseline_mean)

        per_term[term_key] = {
            "baseline_mean": _window_mean(baseline_means),
            "terminal_mean": _window_mean(terminal_means),
            "terminal_shift": _window_mean(worsening),
            "terminal_abs_mean": _window_mean([abs(value) for value in terminal_means]),
        }

    return {
        "avg_terminal_timestep": _window_mean(terminal_timestep_values),
        "term_summary": per_term,
    }


def _build_term_summary(successes: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    term_keys = {
        key
        for episode in [*successes, *failures]
        for key in (episode.get("reward_breakdown", {}) or {}).keys()
        if key != "total"
    }
    summary: dict[str, dict[str, float]] = {}
    for term_key in sorted(term_keys):
        success_values = [_episode_total_for_term(episode, term_key) for episode in successes]
        failure_values = [_episode_total_for_term(episode, term_key) for episode in failures]
        success_terminal = [_terminal_window_total(episode, term_key) for episode in successes]
        failure_terminal = [_terminal_window_total(episode, term_key) for episode in failures]

        success_mean = mean(success_values) if success_values else 0.0
        failure_mean = mean(failure_values) if failure_values else 0.0
        success_terminal_mean = mean(success_terminal) if success_terminal else 0.0
        failure_terminal_mean = mean(failure_terminal) if failure_terminal else 0.0

        summary[term_key] = {
            "success_mean": success_mean,
            "failure_mean": failure_mean,
            "contrast": success_mean - failure_mean,
            "contrast_ratio": (success_mean - failure_mean) / _pooled_scale(success_values, failure_values),
            "success_terminal_mean": success_terminal_mean,
            "failure_terminal_mean": failure_terminal_mean,
            "terminal_contrast": success_terminal_mean - failure_terminal_mean,
            "terminal_contrast_ratio": (success_terminal_mean - failure_terminal_mean) / _pooled_scale(success_terminal, failure_terminal),
        }
    return summary


def _format_term_value(value: float) -> str:
    return f"{value:+.3f}"


def _insight(priority: str, category: str, title: str, body: str, evidence: dict[str, Any], confidence: float) -> dict[str, Any]:
    return {
        "priority": priority,
        "category": category,
        "title": title,
        "body": body,
        "evidence": evidence,
        "confidence": round(max(0.0, min(confidence, 1.0)), 2),
    }


def build_episode_insights(episodes: list[dict[str, Any]], source: str, env_name: str | None = None) -> dict[str, Any]:
    recent = list(episodes or [])[:25]
    successes = [episode for episode in recent if episode.get("episode_outcome") == "success"]
    failures = [episode for episode in recent if episode.get("episode_outcome") == "failure"]
    unknown = [episode for episode in recent if episode.get("episode_outcome") not in {"success", "failure"}]

    summary = {
        "source": source,
        "env_name": env_name,
        "episodes_analyzed": len(recent),
        "success_count": len(successes),
        "failure_count": len(failures),
        "unknown_count": len(unknown),
        "success_rate": (len(successes) / (len(successes) + len(failures))) if (len(successes) + len(failures)) else None,
    }

    result = {
        "summary": summary,
        "term_scores": {},
        "insights": [],
    }

    if len(recent) < 4:
        result["insights"].append(_insight(
            "info",
            "coverage",
            "Not enough episodes yet",
            f"The {source} insight engine needs at least 4 recent episodes before it can compare outcome patterns with confidence.",
            {"episodes_analyzed": len(recent)},
            0.25,
        ))
        return result

    if not successes or not failures:
        dominant_outcome = "success" if successes else "failure"
        dominant_episodes = successes if successes else failures
        single_outcome_summary = _build_single_outcome_terminal_summary(dominant_episodes)
        result["term_scores"] = single_outcome_summary.get("term_summary", {})
        result["insights"].append(_insight(
            "info",
            "coverage",
            f"Only {dominant_outcome} episodes available",
            f"The recent {source} window has only {dominant_outcome} episodes, so contrastive success-vs-failure insights are unavailable. The findings below focus on within-failure terminal patterns instead.",
            {
                "episodes_analyzed": len(recent),
                "success_count": len(successes),
                "failure_count": len(failures),
            },
            0.35,
        ))

        avg_terminal_timestep = single_outcome_summary.get("avg_terminal_timestep", 0.0)
        if avg_terminal_timestep:
            result["insights"].append(_insight(
                "medium",
                "failure-timing" if dominant_outcome == "failure" else "success-timing",
                f"{dominant_outcome.capitalize()} tends to occur around timestep {round(avg_terminal_timestep)}",
                (
                    f"Across the recent {source} window, {dominant_outcome} episodes terminate around timestep {avg_terminal_timestep:.1f} on average. "
                    f"This gives a concrete region to inspect when looking for what reliably precedes the terminal event."
                ),
                {
                    "avg_terminal_timestep": round(avg_terminal_timestep, 3),
                    "episodes_analyzed": len(dominant_episodes),
                },
                0.45,
            ))

        term_summary = single_outcome_summary.get("term_summary", {})
        if term_summary:
            worsening_terms = sorted(
                term_summary.items(),
                key=lambda item: abs(item[1]["terminal_shift"]),
                reverse=True,
            )
            top_worsening = next(
                (
                    (term_key, stats)
                    for term_key, stats in worsening_terms
                    if abs(stats["terminal_shift"]) > 0.01
                ),
                None,
            )
            if top_worsening:
                term_key, stats = top_worsening
                direction = "drops" if stats["terminal_shift"] < 0 else "rises"
                result["insights"].append(_insight(
                    "high" if dominant_outcome == "failure" else "medium",
                    "terminal-pattern",
                    f"`{term_key}` {direction} right before {dominant_outcome}",
                    (
                        f"In these {dominant_outcome} episodes, `{term_key}` moves from an earlier-episode mean of "
                        f"{_format_term_value(stats['baseline_mean'])} to a final-window mean of {_format_term_value(stats['terminal_mean'])}. "
                        f"That {direction} in the last 8 timesteps is one of the strongest repeatable signatures immediately before the terminal event."
                    ),
                    {
                        "term": term_key,
                        "baseline_mean": round(stats["baseline_mean"], 4),
                        "terminal_mean": round(stats["terminal_mean"], 4),
                        "terminal_shift": round(stats["terminal_shift"], 4),
                    },
                    0.5 + min(0.3, abs(stats["terminal_shift"])),
                ))

            top_terminal_magnitude = next(
                (
                    (term_key, stats)
                    for term_key, stats in sorted(
                        term_summary.items(),
                        key=lambda item: item[1]["terminal_abs_mean"],
                        reverse=True,
                    )
                    if stats["terminal_abs_mean"] > 0.01
                ),
                None,
            )
            if top_terminal_magnitude:
                term_key, stats = top_terminal_magnitude
                result["insights"].append(_insight(
                    "medium",
                    "dominant-terminal-term",
                    f"`{term_key}` dominates the last timesteps",
                    (
                        f"Near {dominant_outcome}, `{term_key}` has the largest average terminal-window magnitude at "
                        f"{abs(stats['terminal_mean']):.3f}. That makes it one of the most important terms to inspect when debugging what the agent is optimizing just before the end."
                    ),
                    {
                        "term": term_key,
                        "terminal_mean": round(stats["terminal_mean"], 4),
                        "terminal_abs_mean": round(stats["terminal_abs_mean"], 4),
                    },
                    0.45 + min(0.25, stats["terminal_abs_mean"]),
                ))

        return result

    term_summary = _build_term_summary(successes, failures)
    result["term_scores"] = term_summary
    insights: list[dict[str, Any]] = []

    success_favoring_terms = sorted(
        term_summary.items(),
        key=lambda item: item[1]["contrast_ratio"],
        reverse=True,
    )
    failure_favoring_terms = sorted(
        term_summary.items(),
        key=lambda item: item[1]["contrast_ratio"],
    )
    terminal_failure_terms = sorted(
        term_summary.items(),
        key=lambda item: item[1]["terminal_contrast_ratio"],
    )

    top_success = next(
        ((term_key, stats) for term_key, stats in success_favoring_terms if stats["contrast_ratio"] > 0.35),
        None,
    )
    if top_success:
        term_key, stats = top_success
        insights.append(_insight(
            "high",
            "success-driver",
            f"`{term_key}` tracks with successful episodes",
            (
                f"In the recent {source} window, successful episodes average {_format_term_value(stats['success_mean'])} from `{term_key}` "
                f"versus {_format_term_value(stats['failure_mean'])} in failed episodes. This term is currently one of the strongest "
                f"positive separators between success and failure."
            ),
            {
                "term": term_key,
                "success_mean": round(stats["success_mean"], 4),
                "failure_mean": round(stats["failure_mean"], 4),
                "contrast_ratio": round(stats["contrast_ratio"], 4),
            },
            0.55 + min(0.35, abs(stats["contrast_ratio"]) / 2),
        ))

    top_failure = next(
        ((term_key, stats) for term_key, stats in failure_favoring_terms if stats["contrast_ratio"] < -0.35),
        None,
    )
    if top_failure:
        term_key, stats = top_failure
        insights.append(_insight(
            "high",
            "failure-driver",
            f"`{term_key}` is associated with failure",
            (
                f"Failed episodes average {_format_term_value(stats['failure_mean'])} from `{term_key}` versus "
                f"{_format_term_value(stats['success_mean'])} in successful episodes. This makes `{term_key}` the clearest reward-term "
                f"signal currently aligned with failure."
            ),
            {
                "term": term_key,
                "success_mean": round(stats["success_mean"], 4),
                "failure_mean": round(stats["failure_mean"], 4),
                "contrast_ratio": round(stats["contrast_ratio"], 4),
            },
            0.55 + min(0.35, abs(stats["contrast_ratio"]) / 2),
        ))

    top_terminal_failure = next(
        ((term_key, stats) for term_key, stats in terminal_failure_terms if stats["terminal_contrast_ratio"] < -0.4),
        None,
    )
    if top_terminal_failure:
        term_key, stats = top_terminal_failure
        insights.append(_insight(
            "medium",
            "terminal-pattern",
            f"`{term_key}` worsens near terminal failure",
            (
                f"In the final timesteps, failed episodes accumulate {_format_term_value(stats['failure_terminal_mean'])} from `{term_key}` "
                f"versus {_format_term_value(stats['success_terminal_mean'])} in successful episodes. The terminal window suggests this term "
                f"is a late-episode failure signature rather than just a whole-episode effect."
            ),
            {
                "term": term_key,
                "success_terminal_mean": round(stats["success_terminal_mean"], 4),
                "failure_terminal_mean": round(stats["failure_terminal_mean"], 4),
                "terminal_contrast_ratio": round(stats["terminal_contrast_ratio"], 4),
            },
            0.5 + min(0.3, abs(stats["terminal_contrast_ratio"]) / 2),
        ))

    native_stats = term_summary.get("native")
    if native_stats:
        total_success = mean([_safe_float((episode.get("reward_breakdown", {}) or {}).get("total", episode.get("reward"))) for episode in successes])
        total_failure = mean([_safe_float((episode.get("reward_breakdown", {}) or {}).get("total", episode.get("reward"))) for episode in failures])
        native_gap = abs(native_stats["contrast"] - (total_success - total_failure))
        if native_gap > 0.5:
            insights.append(_insight(
                "medium",
                "shaping-mismatch",
                "Shaped reward is separating outcomes differently than native reward",
                (
                    f"The total reward gap between success and failure is {_format_term_value(total_success - total_failure)}, while the native reward gap "
                    f"is {_format_term_value(native_stats['contrast'])}. That difference suggests the shaped terms are materially changing what the policy is optimizing."
                ),
                {
                    "total_contrast": round(total_success - total_failure, 4),
                    "native_contrast": round(native_stats["contrast"], 4),
                    "gap": round(native_gap, 4),
                },
                0.52 + min(0.28, native_gap / 4),
            ))

    ordered_episodes = sorted(
        [episode for episode in recent if isinstance(episode.get("episode"), (int, float))],
        key=lambda episode: episode.get("episode", 0),
    )
    if len(ordered_episodes) >= 6:
        midpoint = len(ordered_episodes) // 2
        early = ordered_episodes[:midpoint]
        late = ordered_episodes[midpoint:]
        early_success_rate = mean([1.0 if episode.get("episode_outcome") == "success" else 0.0 for episode in early])
        late_success_rate = mean([1.0 if episode.get("episode_outcome") == "success" else 0.0 for episode in late])
        delta = late_success_rate - early_success_rate
        if abs(delta) >= 0.2:
            direction = "improving" if delta > 0 else "degrading"
            insights.append(_insight(
                "medium",
                "trend",
                f"Recent {source} outcome trend is {direction}",
                (
                    f"Success rate moved from {early_success_rate:.0%} in the earlier half of the recent episode window to "
                    f"{late_success_rate:.0%} in the later half. This suggests the current policy behavior is {direction} over time."
                ),
                {
                    "early_success_rate": round(early_success_rate, 4),
                    "late_success_rate": round(late_success_rate, 4),
                    "delta": round(delta, 4),
                },
                0.48 + min(0.3, abs(delta)),
            ))

    if not insights:
        insights.append(_insight(
            "info",
            "baseline",
            "No strong deterministic separator found yet",
            f"The recent {source} episodes do not yet show a single reward term with a large enough success-versus-failure contrast. More episodes or sharper reward shaping may be needed.",
            {"episodes_analyzed": len(recent), "term_count": len(term_summary)},
            0.4,
        ))

    result["insights"] = insights[:6]
    return result
