from __future__ import annotations

import ast
import math
import re
from typing import Any, Callable
import logging
from run_logging import configure_backend_logging

import gymnasium as gym
import numpy as np

from train_backend_reward_tuning.reward_templates import (
    raw_reward_terms_for_env,
    reward_template_for_env,
)

logger = configure_backend_logging(level=logging.INFO)

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
    previous_obs,
    previous_action,
    native_reward: float,
    raw_terms: dict[str, float] | None = None,
    env_name: str | None = None,
    step: int = 0,
    time_sec: float = 0.0,
) -> dict[str, float]:
    context: dict[str, float] = {
        "native": float(native_reward),
        "native_reward": float(native_reward),
        "step": float(step),
        "time_sec": float(time_sec),
    }
    for key, value in (raw_terms or {}).items():
        if _REWARD_KEY_PATTERN.match(key):
            context[key] = float(value)

    # flatten 
    flat_obs = _flat_values(obs) 
    flat_action = _flat_values(action)
    flat_prev_obs = _flat_values(previous_obs, default_length=flat_obs.size)
    flat_prev_action = _flat_values(previous_action, default_length=flat_action.size or 1)

    for index, value in enumerate(flat_obs):
        context[f"obs_{index}"] = float(value)
    for index, value in enumerate(flat_prev_obs):
        context[f"prev_obs_{index}"] = float(value)
    for index, value in enumerate(flat_action):
        context[f"action_{index}"] = float(value)
    for index, value in enumerate(flat_prev_action):
        context[f"prev_action_{index}"] = float(value)

    if env_name:
        for spec in reward_expression_variable_specs(
            env_name=env_name,
            obs_size=int(flat_obs.size),
            action_size=int(flat_action.size or 1),
            raw_term_keys=list((raw_terms or {}).keys()),
        ):
            canonical_name = spec["name"]
            canonical_value = context.get(canonical_name)
            if canonical_value is None:
                continue
            for alias_name in spec.get("aliases", []):
                if alias_name not in context:
                    context[alias_name] = float(canonical_value)

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


def enrich_reward_context_with_task_config(
    context: dict[str, float],
    task_config: dict[str, Any] | None,
) -> dict[str, float]:
    current = dict(context)
    if not task_config:
        return current

    for param in task_config.get("task_params", []) or []:
        key = str(param.get("key") or "").strip()
        if not key or not _REWARD_KEY_PATTERN.match(key):
            continue
        try:
            current[key] = float(param.get("value", 0.0))
        except (TypeError, ValueError):
            current[key] = 0.0

    for signal in task_config.get("derived_signals", []) or []:
        key = str(signal.get("key") or "").strip()
        expression = str(signal.get("expression") or "").strip()
        if not key or not expression or not _REWARD_KEY_PATTERN.match(key):
            continue
        try:
            current[key] = evaluate_reward_expression(expression, current)
        except ValueError:
            current[key] = 0.0

    return current


def reward_expression_variable_names(
    obs_size: int,
    action_size: int,
    *,
    env_name: str | None = None,
    include_previous_obs: bool = True,
    include_previous_action: bool = True,
    raw_term_keys: list[str] | None = None,
) -> list[str]:
    variable_names: list[str] = []
    for spec in reward_expression_variable_specs(
        env_name=env_name,
        obs_size=obs_size, # flattened obs
        action_size=action_size, # flattened action
        include_previous_obs=include_previous_obs,
        include_previous_action=include_previous_action,
        raw_term_keys=raw_term_keys,
    ):
        variable_names.append(spec["name"])
        variable_names.extend(spec.get("aliases", []))
    deduped: list[str] = []
    seen: set[str] = set()
    for variable_name in variable_names:
        if variable_name not in seen:
            seen.add(variable_name)
            deduped.append(variable_name)
    return deduped


def reward_expression_variable_specs(
    env_name: str | None,
    obs_size: int,
    action_size: int,
    *,
    include_previous_obs: bool = True,
    include_previous_action: bool = True,
    raw_term_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    # The following specs are always available.
    # The variables are 
    # - step: the current episode timestep, starting at 1
    # - time_sec: the approximate elapsed episode time in seconds
    # - native: the native reward returned by the environment
    """
    This code snippet defines a function called `reward_expression_variable_specs` that generates a list of dictionaries representing variable specifications for a reward expression in a reinforcement learning environment. The function takes several parameters including `env_name`, `obs_size`, and `action_size`, and has optional parameters like `include_previous_action` and `raw_term_keys`. 

    The function starts by initializing a list called `specs` with three dictionaries representing the `step`, `time_sec`, and `native` variables. It then calls another function `_reward_observation_specs` to get a set of observation specifications based on the `env_name` and `obs_size`. 

    The function then iterates over the range of `obs_size`, checking if each observation specification exists in the set. If it does, it extracts the display name and aliases from the specification and appends them to the `specs` list. If it doesn't exist, it adds a new dictionary representing the observation component to the `specs` list.

    Next, the function iterates over the range of `action_size`, adding dictionaries representing the action components to the `specs` list. It also checks if `include_previous_action` is `True`, and if so, adds dictionaries representing the previous action components to the `specs` list.

    Finally, the function iterates over the `raw_term_keys` list (if provided) and adds dictionaries representing the raw reward features to the `specs` list.

    The function returns the `specs` list containing the variable specifications.

    """
    specs: list[dict[str, Any]] = [
        {
            "name": "step",
            "source": "runtime",
            "display_name": "step",
            "description": "Current episode timestep starting at 1.",
            "aliases": [],
        },
        {
            "name": "time_sec",
            "source": "runtime",
            "display_name": "time_sec",
            "description": "Approximate elapsed episode time in seconds.",
            "aliases": [],
        },
        {
            "name": "native",
            "source": "reward",
            "display_name": "native_reward",
            "description": "Native reward returned by the environment.",
            "aliases": ["native_reward"],
        }
    ]

    # Add observation variables
    # Add the aliases from the observation specs if available, otherwise add generic obs_i variables.

    # it takes a description of every observation component from the environment-specific function _reward_observation_specs, which returns a dictionary mapping observation indices to their specifications. For each observation index up to obs_size, it checks if there is a corresponding specification in obs_specs. If there is, it extracts the display name and aliases from the specification and adds them to the specs list. If there isn't, it adds a generic specification for that observation index with a default display name and description.
    obs_specs = _reward_observation_specs(env_name or "", obs_size)
    for index in range(max(0, obs_size)):
        obs_spec = obs_specs.get(index)

        if obs_spec is not None:
            alias_values = [obs_spec.get("display_name"), *obs_spec.get("aliases", [])]
            deduped_aliases: list[str] = []
            for alias_name in alias_values:
                alias_value = str(alias_name or "").strip()
                if alias_value and alias_value != obs_spec["name"] and alias_value not in deduped_aliases:
                    deduped_aliases.append(alias_value)
            obs_spec["aliases"] = deduped_aliases
            specs.append(obs_spec)
            continue
        specs.append(
            {
                "name": f"obs_{index}",
                "source": "observation",
                "display_name": f"observation_{index}",
                "description": f"Flattened observation component {index}.",
                "aliases": [f"observation_{index}"],
            }
        )

    if include_previous_obs:
        for index in range(max(0, obs_size)):
            obs_spec = obs_specs.get(index)
            aliases: list[str] = []
            if obs_spec is not None:
                display_name = str(obs_spec.get("display_name") or "").strip()
                raw_aliases = [display_name, *obs_spec.get("aliases", [])]
                for alias_name in raw_aliases:
                    alias_value = str(alias_name or "").strip()
                    if alias_value:
                        prefixed = f"prev_{alias_value}"
                        if prefixed not in aliases:
                            aliases.append(prefixed)
            else:
                aliases.append(f"previous_observation_{index}")
            """
            Adding a specification component for the previous observation environment.
            """
            specs.append(
                {
                    "name": f"prev_obs_{index}",
                    "source": "observation",
                    "display_name": aliases[0] if aliases else f"previous_observation_{index}",
                    "description": f"Previous flattened observation component {index}.",
                    "aliases": aliases,
                }
            )

    # Add action variables as observations. 
    for index in range(max(1, action_size)):
        action_aliases = [f"current_action_{index}"]
        if action_size == 1 and index == 0:
            action_aliases.extend(["action", "current_action"])
        specs.append(
            {
                "name": f"action_{index}",
                "source": "action",
                "display_name": action_aliases[0],
                "description": f"Current action component {index}.",
                "aliases": [action_aliases[0], *action_aliases[1:]],
            }
        )
    # Add previous action variables as observations
    if include_previous_action:
        for index in range(max(1, action_size)):
            previous_aliases = [f"previous_action_{index}"]
            if action_size == 1 and index == 0:
                previous_aliases.extend(["prev_action", "previous_action"])
            specs.append(
                {
                    "name": f"prev_action_{index}",
                    "source": "action",
                    "display_name": previous_aliases[0],
                    "description": f"Previous action component {index}.",
                    "aliases": [previous_aliases[0], *previous_aliases[1:]],
                }
            )
    for key in raw_term_keys or []:
        if key == "native":
            continue
        specs.append(
            {
                "name": key,
                "source": "reward_term",
                "display_name": key,
                "description": f"Built-in raw reward feature '{key}'.",
                "aliases": [],
            }
        )
    
    # to see what the signals are
    logger.info("Specs: %s", specs)
    return specs


def _reward_observation_specs(env_name: str, obs_size: int) -> dict[int, dict[str, Any]]:
    """
    Generate dictionary of observation specifications for a variety of environments

    Two parameters taken: env_name and obs_size
        - env_name is used to determine the prefix of the environment (e.g., "cartpole" from "CartPole-v1") 
        and select the appropriate set of observation specifications based on that prefix.
        - obs_size is used to determine how many observation components to generate specifications for. 
        The function iterates from 0 to obs_size-1 and checks if there are predefined specifications for 
        each index based on the environment prefix. If there are, it uses those specifications; otherwise, 
        it generates generic specifications for any remaining indices.
    
    The function returns a dictionary mapping observation indices to their specifications, which include the name, source, display name, description, and aliases for each observation component.
    """
    prefix = (env_name or "").split("/", 1)[-1].split("-", 1)[0].lower()

    """
    Create tuple of index and dictionary of observation specifications using the make_spec function
    """
    def make_spec(index: int, display_name: str, description: str, *aliases: str) -> tuple[int, dict[str, Any]]:
        clean_aliases: list[str] = []
        for alias_name in aliases:
            alias_value = str(alias_name).strip()
            if alias_value and alias_value not in clean_aliases:
                clean_aliases.append(alias_value)
        return (
            index,
            {
                "name": f"obs_{index}",
                "source": "observation",
                "display_name": display_name,
                "description": description,
                "aliases": clean_aliases,
            },
        )

    entries: list[tuple[int, dict[str, Any]]] = []
    if prefix in {"cartpole", "cartpoleloose"}:
        entries = [
            make_spec(0, "cart_position", "Cart position along the track.", "x", "cart_x"),
            make_spec(1, "cart_velocity", "Cart velocity along the track.", "x_dot", "cart_x_velocity"),
            make_spec(2, "pole_angle", "Pole angle in radians; left/right sway target lives here.", "theta"),
            make_spec(3, "pole_velocity", "Pole angular velocity in radians per second.", "theta_dot", "pole_angular_velocity"),
        ]
    elif prefix in {"mountaincar", "mountaincarcontinuous"}:
        entries = [
            make_spec(0, "position", "Car position along the hill.", "x"),
            make_spec(1, "velocity", "Car velocity along the hill.", "x_dot"),
        ]
    elif prefix == "pendulum":
        entries = [
            make_spec(0, "x", "Pendulum x component; equals cos(theta).", "cos_theta"),
            make_spec(1, "y", "Pendulum y component; equals sin(theta).", "sin_theta"),
            make_spec(2, "angular_velocity", "Pendulum angular velocity.", "theta_dot"),
        ]
    elif prefix == "acrobot":
        entries = [
            make_spec(0, "cos_theta1", "Cosine of the first joint angle.", "link1_x"),
            make_spec(1, "sin_theta1", "Sine of the first joint angle.", "link1_y"),
            make_spec(2, "cos_theta2", "Cosine of the second joint angle.", "link2_x"),
            make_spec(3, "sin_theta2", "Sine of the second joint angle.", "link2_y"),
            make_spec(4, "joint1_velocity", "Angular velocity of the first joint.", "theta1_dot"),
            make_spec(5, "joint2_velocity", "Angular velocity of the second joint.", "theta2_dot"),
        ]
    elif prefix == "lunarlander":
        entries = [
            make_spec(0, "x", "Lander horizontal position.", "horizontal_position"),
            make_spec(1, "y", "Lander vertical position.", "vertical_position"),
            make_spec(2, "x_velocity", "Lander horizontal velocity.", "vx"),
            make_spec(3, "y_velocity", "Lander vertical velocity.", "vy"),
            make_spec(4, "angle", "Lander rotation angle in radians.", "theta"),
            make_spec(5, "angular_velocity", "Lander angular velocity.", "theta_dot"),
            make_spec(6, "left_leg_contact", "Left leg contact flag."),
            make_spec(7, "right_leg_contact", "Right leg contact flag."),
        ]
    elif prefix == "invertedpendulum":
        entries = [
            make_spec(0, "cart_position", "Cart position along the rail.", "x", "cart_x"),
            make_spec(1, "pole_angle", "Pendulum angle in radians.", "theta"),
            make_spec(2, "cart_velocity", "Cart velocity along the rail.", "x_dot", "cart_x_velocity"),
            make_spec(3, "pole_velocity", "Pendulum angular velocity.", "theta_dot", "pole_angular_velocity"),
        ]
    elif prefix == "inverteddoublependulum":
        entries = [
            make_spec(0, "cart_position", "Cart position along the rail.", "x", "cart_x"),
            make_spec(1, "pole_angle_1", "First pendulum angle in radians.", "theta1"),
            make_spec(2, "pole_angle_2", "Second pendulum angle in radians.", "theta2"),
            make_spec(3, "cart_velocity", "Cart velocity along the rail.", "x_dot", "cart_x_velocity"),
            make_spec(4, "pole_velocity_1", "First pendulum angular velocity.", "theta1_dot"),
            make_spec(5, "pole_velocity_2", "Second pendulum angular velocity.", "theta2_dot"),
        ]
    elif prefix == "reacher":
        entries = [
            make_spec(0, "cos_theta1", "Cosine of the first arm joint angle."),
            make_spec(1, "cos_theta2", "Cosine of the second arm joint angle."),
            make_spec(2, "sin_theta1", "Sine of the first arm joint angle."),
            make_spec(3, "sin_theta2", "Sine of the second arm joint angle."),
            make_spec(4, "target_x", "Target x position."),
            make_spec(5, "target_y", "Target y position."),
            make_spec(6, "joint1_velocity", "First arm joint velocity.", "theta1_dot"),
            make_spec(7, "joint2_velocity", "Second arm joint velocity.", "theta2_dot"),
            make_spec(8, "fingertip_delta_x", "Fingertip minus target x offset."),
            make_spec(9, "fingertip_delta_y", "Fingertip minus target y offset."),
            make_spec(10, "fingertip_delta_z", "Fingertip minus target z offset."),
        ]
    elif prefix == "halfcheetah":
        entries = _build_named_specs(
            [
                ("z", "Root body height."),
                ("torso_pitch", "Root torso pitch angle."),
                ("back_thigh_angle", "Back thigh joint angle."),
                ("back_shin_angle", "Back shin joint angle."),
                ("back_foot_angle", "Back foot joint angle."),
                ("front_thigh_angle", "Front thigh joint angle."),
                ("front_shin_angle", "Front shin joint angle."),
                ("front_foot_angle", "Front foot joint angle."),
            ],
            [
                ("x_velocity", "Forward velocity."),
                ("z_velocity", "Vertical velocity."),
                ("pitch_velocity", "Torso pitch angular velocity."),
                ("back_thigh_velocity", "Back thigh joint velocity."),
                ("back_shin_velocity", "Back shin joint velocity."),
                ("back_foot_velocity", "Back foot joint velocity."),
                ("front_thigh_velocity", "Front thigh joint velocity."),
                ("front_shin_velocity", "Front shin joint velocity."),
                ("front_foot_velocity", "Front foot joint velocity."),
            ],
        )
    elif prefix == "hopper":
        entries = _build_named_specs(
            [
                ("z", "Torso height."),
                ("torso_angle", "Torso pitch angle."),
                ("thigh_angle", "Thigh joint angle."),
                ("leg_angle", "Leg joint angle."),
                ("foot_angle", "Foot joint angle."),
            ],
            [
                ("x_velocity", "Forward velocity."),
                ("z_velocity", "Vertical velocity."),
                ("torso_angular_velocity", "Torso angular velocity."),
                ("thigh_velocity", "Thigh joint velocity."),
                ("leg_velocity", "Leg joint velocity."),
                ("foot_velocity", "Foot joint velocity."),
            ],
        )
    elif prefix == "walker2d":
        entries = _build_named_specs(
            [
                ("z", "Torso height."),
                ("torso_angle", "Torso pitch angle."),
                ("right_thigh_angle", "Right thigh joint angle."),
                ("right_leg_angle", "Right leg joint angle."),
                ("right_foot_angle", "Right foot joint angle."),
                ("left_thigh_angle", "Left thigh joint angle."),
                ("left_leg_angle", "Left leg joint angle."),
                ("left_foot_angle", "Left foot joint angle."),
            ],
            [
                ("x_velocity", "Forward velocity."),
                ("z_velocity", "Vertical velocity."),
                ("torso_angular_velocity", "Torso angular velocity."),
                ("right_thigh_velocity", "Right thigh joint velocity."),
                ("right_leg_velocity", "Right leg joint velocity."),
                ("right_foot_velocity", "Right foot joint velocity."),
                ("left_thigh_velocity", "Left thigh joint velocity."),
                ("left_leg_velocity", "Left leg joint velocity."),
                ("left_foot_velocity", "Left foot joint velocity."),
            ],
        )
    elif prefix == "swimmer":
        entries = _build_named_specs(
            [
                ("joint_angle_0", "First body joint angle."),
                ("joint_angle_1", "Second body joint angle."),
                ("joint_angle_2", "Third body joint angle."),
                ("joint_angle_3", "Fourth body joint angle."),
                ("joint_angle_4", "Fifth body joint angle."),
            ],
            [
                ("x_velocity", "Forward velocity."),
                ("y_velocity", "Sideways velocity."),
                ("joint_velocity_0", "First joint velocity."),
                ("joint_velocity_1", "Second joint velocity."),
                ("joint_velocity_2", "Third joint velocity."),
                ("joint_velocity_3", "Fourth joint velocity."),
                ("joint_velocity_4", "Fifth joint velocity."),
            ],
        )
    elif prefix == "ant":
        entries = _build_named_specs(
            [
                ("z", "Root body height."),
                ("quat_w", "Root body orientation quaternion w."),
                ("quat_x", "Root body orientation quaternion x."),
                ("quat_y", "Root body orientation quaternion y."),
                ("quat_z", "Root body orientation quaternion z."),
                ("joint_pos_0", "Ant joint position 0."),
                ("joint_pos_1", "Ant joint position 1."),
                ("joint_pos_2", "Ant joint position 2."),
                ("joint_pos_3", "Ant joint position 3."),
                ("joint_pos_4", "Ant joint position 4."),
                ("joint_pos_5", "Ant joint position 5."),
                ("joint_pos_6", "Ant joint position 6."),
                ("joint_pos_7", "Ant joint position 7."),
            ],
            [
                ("x_velocity", "Root body x velocity."),
                ("y_velocity", "Root body y velocity."),
                ("z_velocity", "Root body z velocity."),
                ("roll_velocity", "Root body roll angular velocity."),
                ("pitch_velocity", "Root body pitch angular velocity."),
                ("yaw_velocity", "Root body yaw angular velocity."),
                ("joint_velocity_0", "Ant joint velocity 0."),
                ("joint_velocity_1", "Ant joint velocity 1."),
                ("joint_velocity_2", "Ant joint velocity 2."),
                ("joint_velocity_3", "Ant joint velocity 3."),
                ("joint_velocity_4", "Ant joint velocity 4."),
                ("joint_velocity_5", "Ant joint velocity 5."),
                ("joint_velocity_6", "Ant joint velocity 6."),
                ("joint_velocity_7", "Ant joint velocity 7."),
            ],
        )
    elif prefix == "humanoid":
        root_position_specs = [
            ("z", "Root body height."),
            ("quat_w", "Root body orientation quaternion w."),
            ("quat_x", "Root body orientation quaternion x."),
            ("quat_y", "Root body orientation quaternion y."),
            ("quat_z", "Root body orientation quaternion z."),
        ]
        root_velocity_specs = [
            ("x_velocity", "Root body x velocity."),
            ("y_velocity", "Root body y velocity."),
            ("z_velocity", "Root body z velocity."),
            ("roll_velocity", "Root body roll angular velocity."),
            ("pitch_velocity", "Root body pitch angular velocity."),
            ("yaw_velocity", "Root body yaw angular velocity."),
        ]
        entries = _build_filled_specs(
            obs_size,
            root_position_specs,
            root_velocity_specs,
            extra_prefixes=("joint_pos", "joint_vel", "cinert", "cvel", "qfrc_actuator", "cfrc_ext"),
        )

    return dict(entries)


def _build_named_specs(
    position_specs: list[tuple[str, str]],
    velocity_specs: list[tuple[str, str]],
) -> list[tuple[int, dict[str, Any]]]:
    entries: list[tuple[int, dict[str, Any]]] = []
    for index, (display_name, description) in enumerate(position_specs):
        entries.append(
            (
                index,
                {
                    "name": f"obs_{index}",
                    "source": "observation",
                    "display_name": display_name,
                    "description": description,
                    "aliases": [display_name],
                },
            )
        )
    offset = len(position_specs)
    for index, (display_name, description) in enumerate(velocity_specs, start=offset):
        entries.append(
            (
                index,
                {
                    "name": f"obs_{index}",
                    "source": "observation",
                    "display_name": display_name,
                    "description": description,
                    "aliases": [display_name],
                },
            )
        )
    return entries


def _build_filled_specs(
    obs_size: int,
    position_specs: list[tuple[str, str]],
    velocity_specs: list[tuple[str, str]],
    *,
    extra_prefixes: tuple[str, ...],
) -> list[tuple[int, dict[str, Any]]]:
    entries = _build_named_specs(position_specs, velocity_specs)
    next_index = len(entries)
    if next_index >= obs_size:
        return entries[:obs_size]
    prefix_index = 0
    while next_index < obs_size:
        prefix = extra_prefixes[min(prefix_index, len(extra_prefixes) - 1)]
        display_name = f"{prefix}_{next_index - len(position_specs) - len(velocity_specs)}"
        entries.append(
            (
                next_index,
                {
                    "name": f"obs_{next_index}",
                    "source": "observation",
                    "display_name": display_name,
                    "description": f"Additional observation feature '{display_name}'.",
                    "aliases": [display_name],
                },
            )
        )
        next_index += 1
        if prefix_index < len(extra_prefixes) - 1 and (next_index - len(position_specs) - len(velocity_specs)) % 10 == 0:
            prefix_index += 1
    return entries


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
        task_config_provider: Callable[[], dict[str, Any] | None] | None = None,
    ) -> None:
        super().__init__(env)
        self.env_name = env_name
        self.config_provider = config_provider
        self.task_config_provider = task_config_provider or (lambda: None)
        self._previous_obs = None
        self._previous_action = None
        self._episode_step = 0
        env_unwrapped = getattr(self.env, "unwrapped", self.env)
        base_dt = getattr(env_unwrapped, "dt", None)
        if base_dt is None:
            tau = getattr(env_unwrapped, "tau", None)
            base_dt = tau if tau is not None else 1.0
        self._step_duration = float(base_dt)
        self._reset_episode_sums()

    def _reset_episode_sums(self) -> None:
        self._episode_term_sums = {
            term["key"]: 0.0 for term in reward_template_for_env(self.env_name)
        }
        self._episode_reward_history: list[dict[str, Any]] = []
        self._episode_behavior_trace: list[dict[str, Any]] = []

    def reset(self, **kwargs):
        self._previous_obs = None
        self._previous_action = None
        self._episode_step = 0
        self._reset_episode_sums()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, native_reward, terminated, truncated, info = self.env.step(action)
        self._episode_step += 1
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
            previous_obs=self._previous_obs,
            previous_action=self._previous_action,
            native_reward=native_reward,
            raw_terms=raw_terms,
            env_name=self.env_name,
            step=self._episode_step,
            time_sec=self._episode_step * self._step_duration,
        )
        expression_context = enrich_reward_context_with_task_config(
            expression_context,
            self.task_config_provider(),
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
                "time_sec": float(self._episode_step * self._step_duration),
                "reward": float(total_reward),
                "reward_breakdown": {"total": float(total_reward), **reward_breakdown},
                "reward_raw_terms": {key: float(value) for key, value in raw_terms.items()},
            }
        )
        self._episode_behavior_trace.append(
            {
                "step": len(self._episode_behavior_trace) + 1,
                "time_sec": float(self._episode_step * self._step_duration),
                "observation": np.asarray(obs, dtype=np.float32).tolist(),
                "previous_observation": np.asarray(
                    self._previous_obs if self._previous_obs is not None else np.zeros_like(np.asarray(obs, dtype=np.float32)),
                    dtype=np.float32,
                ).tolist(),
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "previous_action": np.asarray(self._previous_action if self._previous_action is not None else np.zeros_like(np.asarray(action, dtype=np.float32)), dtype=np.float32).tolist(),
            }
        )

        self._previous_obs = np.asarray(obs, dtype=np.float32).copy()
        self._previous_action = np.asarray(action, dtype=np.float32).copy()

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
            info["behavior_trace_episode"] = list(self._episode_behavior_trace)
            info["episode_outcome"] = episode_outcome
            info["episode_outcome_reason"] = outcome_reason
            info["episode_terminal_timestep"] = len(self._episode_reward_history)
            info["episode_terminated"] = bool(terminated)
            info["episode_truncated"] = bool(truncated)

        self._previous_action = action
        return obs, float(total_reward), terminated, truncated, info
