from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym

from train_backend_reward_tuning.reward_templates import (
    raw_reward_terms_for_env,
    reward_template_for_env,
)


def reward_monitor_keys(env_name: str) -> tuple[str, ...]:
    keys = [f"reward_{term['key']}" for term in reward_template_for_env(env_name)]
    keys.append("reward_total")
    return tuple(keys)


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

        if terminated or truncated:
            episode_total = float(sum(self._episode_term_sums.values()))
            info["reward_total"] = episode_total
            for key, value in self._episode_term_sums.items():
                info[f"reward_{key}"] = float(value)
            info["reward_breakdown_episode"] = {
                "total": episode_total,
                **{key: float(value) for key, value in self._episode_term_sums.items()},
            }

        self._previous_action = action
        return obs, float(total_reward), terminated, truncated, info
