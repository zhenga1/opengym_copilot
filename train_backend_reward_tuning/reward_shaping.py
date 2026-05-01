from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import gymnasium as gym
import numpy as np


DEFAULT_REWARD_TEMPLATE = [
    {
        "key": "native",
        "label": "Native Reward",
        "description": "Original reward returned by the Gym environment.",
        "weight": 1.0,
        "enabled": True,
    }
]

_CARTPOLE_TEMPLATE = [
    {
        "key": "native",
        "label": "Native Reward",
        "description": "Original CartPole reward from Gym.",
        "weight": 1.0,
        "enabled": True,
    },
    {
        "key": "survival_bonus",
        "label": "Survival Bonus",
        "description": "Extra constant reward per step while the pole survives.",
        "weight": 0.0,
        "enabled": True,
    },
    {
        "key": "cart_position_penalty",
        "label": "Cart Position Penalty",
        "description": "Penalty based on squared cart position from center.",
        "weight": 0.0,
        "enabled": True,
    },
    {
        "key": "cart_velocity_penalty",
        "label": "Cart Velocity Penalty",
        "description": "Penalty based on squared cart velocity.",
        "weight": 0.0,
        "enabled": True,
    },
    {
        "key": "pole_angle_penalty",
        "label": "Pole Angle Penalty",
        "description": "Penalty based on squared pole angle.",
        "weight": 0.0,
        "enabled": True,
    },
    {
        "key": "pole_velocity_penalty",
        "label": "Pole Velocity Penalty",
        "description": "Penalty based on squared pole angular velocity.",
        "weight": 0.0,
        "enabled": True,
    },
    {
        "key": "action_change_penalty",
        "label": "Action Change Penalty",
        "description": "Penalty when the action flips from the previous step.",
        "weight": 0.0,
        "enabled": True,
    },
]


def reward_template_for_env(env_name: str) -> list[dict[str, Any]]:
    if env_name.startswith("CartPole-"):
        return deepcopy(_CARTPOLE_TEMPLATE)
    return deepcopy(DEFAULT_REWARD_TEMPLATE)


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
        self._previous_action: int | float | None = None
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
        raw_terms = self._raw_terms(obs=obs, action=action, native_reward=native_reward)
        terms = normalize_reward_terms(self.env_name, self.config_provider())

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
            info["reward_total"] = float(total_reward if not reward_breakdown else sum(self._episode_term_sums.values()))
            for key, value in self._episode_term_sums.items():
                info[f"reward_{key}"] = float(value)
            info["reward_breakdown_episode"] = {
                "total": float(sum(self._episode_term_sums.values())),
                **{key: float(value) for key, value in self._episode_term_sums.items()},
            }

        self._previous_action = self._action_scalar(action)
        return obs, float(total_reward), terminated, truncated, info

    def _raw_terms(self, obs, action, native_reward: float) -> dict[str, float]:
        terms = {"native": float(native_reward)}
        if not self.env_name.startswith("CartPole-"):
            return terms

        flat_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if flat_obs.shape[0] < 4:
            return terms

        x, x_dot, theta, theta_dot = flat_obs[:4]
        current_action = self._action_scalar(action)
        action_changed = 0.0
        if self._previous_action is not None and current_action is not None:
            action_changed = -1.0 if int(current_action) != int(self._previous_action) else 0.0

        terms.update(
            {
                "survival_bonus": 1.0,
                "cart_position_penalty": -float(x * x),
                "cart_velocity_penalty": -float(x_dot * x_dot),
                "pole_angle_penalty": -float(theta * theta),
                "pole_velocity_penalty": -float(theta_dot * theta_dot),
                "action_change_penalty": float(action_changed),
            }
        )
        return terms

    @staticmethod
    def _action_scalar(action) -> int | float | None:
        if action is None:
            return None
        arr = np.asarray(action).reshape(-1)
        if arr.size == 0:
            return None
        value = arr[0]
        if np.issubdtype(arr.dtype, np.integer):
            return int(value)
        return float(value)
