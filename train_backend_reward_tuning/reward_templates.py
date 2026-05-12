from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


def _term(
    key: str,
    label: str,
    description: str,
    weight: float,
    enabled: bool = True,
    expression: str = "",
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "description": description,
        "weight": float(weight),
        "enabled": enabled,
        "expression": str(expression or "").strip(),
    }


DEFAULT_REWARD_TEMPLATE = [
    _term("native", "Native Reward", "Original reward returned by the Gym environment.", 1.0, expression="native_reward"),
    _term("action_magnitude_penalty", "Action Magnitude Penalty", "Penalty on large actions to reduce control effort.", 0.0, expression="-square(action_0)"),
    _term("action_change_penalty", "Action Change Penalty", "Penalty on abrupt action changes between steps.", 0.0, expression="-abs(action_0 - prev_action_0)"),
]

_CARTPOLE_TEMPLATE = [
    _term("native", "Native Reward", "Original CartPole reward from Gym.", 1.0, expression="native_reward"),
    _term("survival_bonus", "Survival Bonus", "Extra reward for each stable step.", 0.2, expression="1.0"),
    _term("cart_position_penalty", "Cart Position Penalty", "Penalty for moving away from the track center.", 0.6, expression="-square(cart_position)"),
    _term("cart_velocity_penalty", "Cart Velocity Penalty", "Penalty for large cart velocity.", 0.05, expression="-square(cart_velocity)"),
    _term("pole_angle_penalty", "Pole Angle Penalty", "Penalty for pole tilt.", 1.0, expression="-square(pole_angle)"),
    _term("pole_velocity_penalty", "Pole Velocity Penalty", "Penalty for fast pole rotation.", 0.1, expression="-square(pole_velocity)"),
    _term("action_change_penalty", "Action Change Penalty", "Penalty for rapidly flipping actions.", 0.05, expression="-abs(action - prev_action)"),
]

_MOUNTAIN_CAR_TEMPLATE = [
    _term("native", "Native Reward", "Original MountainCar reward from Gym.", 1.0, expression="native_reward"),
    _term("hill_progress_bonus", "Hill Progress Bonus", "Reward for moving toward the right hill and goal.", 1.0, expression="position + 0.5"),
    _term("speed_bonus", "Speed Bonus", "Reward for building momentum.", 0.25, expression="abs(velocity)"),
    _term("goal_side_bonus", "Goal Side Bonus", "Reward for spending time on the goal side of the valley.", 0.5, expression="max(position, 0.0)"),
    _term("action_change_penalty", "Action Change Penalty", "Penalty for rapidly alternating throttle direction.", 0.05, expression="-abs(action - prev_action)"),
]

_MOUNTAIN_CAR_CONTINUOUS_TEMPLATE = [
    _term("native", "Native Reward", "Original continuous MountainCar reward.", 1.0, expression="native_reward"),
    _term("hill_progress_bonus", "Hill Progress Bonus", "Reward for climbing toward the flag.", 1.0, expression="position + 0.5"),
    _term("speed_bonus", "Speed Bonus", "Reward for useful momentum.", 0.2, expression="abs(velocity)"),
    _term("throttle_penalty", "Throttle Penalty", "Penalty on large continuous throttle.", 0.1, expression="-square(action)"),
    _term("action_change_penalty", "Action Change Penalty", "Penalty on sudden throttle changes.", 0.05, expression="-abs(action - prev_action)"),
]

_ACROBOT_TEMPLATE = [
    _term("native", "Native Reward", "Original Acrobot reward from Gym.", 1.0, expression="native_reward"),
    _term("tip_height_bonus", "Tip Height Bonus", "Reward for raising the tip upward.", 1.2, expression="-cos_theta1 - cos_theta1 * cos_theta2 + sin_theta1 * sin_theta2"),
    _term("swing_momentum_bonus", "Swing Momentum Bonus", "Reward for useful angular motion.", 0.15, expression="abs(joint1_velocity) + abs(joint2_velocity)"),
    _term("joint_velocity_penalty", "Joint Velocity Penalty", "Penalty on excessive joint speed.", 0.08, expression="-square(joint1_velocity) - square(joint2_velocity)"),
    _term("action_change_penalty", "Action Change Penalty", "Penalty for chattering torque commands.", 0.05, expression="-abs(action - prev_action)"),
]

_PENDULUM_TEMPLATE = [
    _term("native", "Native Reward", "Original Pendulum reward from Gym.", 1.0, expression="native_reward"),
    _term("upright_bonus", "Upright Bonus", "Reward for keeping the pendulum upright.", 1.0, expression="x"),
    _term("angular_velocity_penalty", "Angular Velocity Penalty", "Penalty for spinning too fast.", 0.1, expression="-square(angular_velocity)"),
    _term("torque_penalty", "Torque Penalty", "Penalty on high torque usage.", 0.08, expression="-square(action)"),
    _term("action_change_penalty", "Action Change Penalty", "Penalty for abrupt torque swings.", 0.04, expression="-abs(action - prev_action)"),
]

_LUNAR_LANDER_TEMPLATE = [
    _term("native", "Native Reward", "Original LunarLander reward from Gym.", 1.0, expression="native_reward"),
    _term("centering_penalty", "Centering Penalty", "Penalty for horizontal drift from pad center.", 0.35, expression="-square(x)"),
    _term("landing_speed_penalty", "Landing Speed Penalty", "Penalty for large landing velocity.", 0.3, expression="-square(x_velocity) - square(y_velocity)"),
    _term("angle_penalty", "Angle Penalty", "Penalty for tilted body angle.", 0.25, expression="-square(angle)"),
    _term("leg_contact_bonus", "Leg Contact Bonus", "Reward for stable ground contact.", 0.4, expression="left_leg_contact + right_leg_contact"),
    _term("fuel_penalty", "Fuel Penalty", "Penalty on large engine commands.", 0.08, expression="-square(action_0) - square(action_1)"),
]

_LOCOMOTION_TEMPLATE = [
    _term("native", "Native Reward", "Original locomotion reward from Gym.", 1.0, expression="native_reward"),
    _term("forward_bonus", "Forward Bonus", "Reward for forward movement.", 1.0, expression="forward_bonus"),
    _term("healthy_bonus", "Healthy Bonus", "Reward for staying alive or upright.", 0.5, expression="healthy_bonus"),
    _term("control_penalty", "Control Penalty", "Penalty on large control effort.", 1.0, expression="control_penalty"),
    _term("contact_penalty", "Contact Penalty", "Penalty on harsh contact or impacts.", 1.0, expression="contact_penalty"),
    _term("stability_penalty", "Stability Penalty", "Penalty for unstable posture or oscillation.", 0.08, expression="stability_penalty"),
    _term("action_change_penalty", "Action Change Penalty", "Penalty for abrupt control changes.", 0.04, expression="-abs(action_0 - prev_action_0)"),
]

_HUMANOID_STANDUP_TEMPLATE = [
    _term("native", "Native Reward", "Original HumanoidStandup reward from Gym.", 1.0, expression="native_reward"),
    _term("standup_bonus", "Standup Bonus", "Reward for lifting the torso upward.", 1.0, expression="standup_bonus"),
    _term("control_penalty", "Control Penalty", "Penalty on large actuator effort.", 1.0, expression="control_penalty"),
    _term("contact_penalty", "Contact Penalty", "Penalty on hard impacts.", 1.0, expression="contact_penalty"),
    _term("balance_penalty", "Balance Penalty", "Penalty for unstable posture while standing.", 0.1, expression="balance_penalty"),
    _term("action_change_penalty", "Action Change Penalty", "Penalty for jerky control sequences.", 0.04, expression="-abs(action_0 - prev_action_0)"),
]

_REACHER_TEMPLATE = [
    _term("native", "Native Reward", "Original Reacher reward from Gym.", 1.0, expression="native_reward"),
    _term("target_proximity_bonus", "Target Proximity Bonus", "Reward for moving the fingertip closer to the target.", 1.0, expression="sqrt(square(fingertip_delta_x) + square(fingertip_delta_y) + square(fingertip_delta_z)) * -1"),
    _term("distance_penalty", "Distance Penalty", "Penalty for target distance.", 1.0, expression="-target_proximity_bonus"),
    _term("control_penalty", "Control Penalty", "Penalty on control effort.", 1.0, expression="-square(action_0) - square(action_1)"),
    _term("action_change_penalty", "Action Change Penalty", "Penalty for abrupt motor changes.", 0.05, expression="-abs(action_0 - prev_action_0) - abs(action_1 - prev_action_1)"),
]

_PUSHER_TEMPLATE = [
    _term("native", "Native Reward", "Original Pusher reward from Gym.", 1.0, expression="native_reward"),
    _term("object_to_goal_bonus", "Object To Goal Bonus", "Reward for moving the object toward the goal.", 1.0, expression="object_to_goal_bonus"),
    _term("hand_to_object_bonus", "Hand To Object Bonus", "Reward for keeping the hand near the object.", 0.5, expression="hand_to_object_bonus"),
    _term("control_penalty", "Control Penalty", "Penalty on large control effort.", 1.0, expression="control_penalty"),
    _term("action_change_penalty", "Action Change Penalty", "Penalty for abrupt motor changes.", 0.05, expression="-abs(action_0 - prev_action_0)"),
]

_INVERTED_PENDULUM_TEMPLATE = [
    _term("native", "Native Reward", "Original InvertedPendulum reward from Gym.", 1.0, expression="native_reward"),
    _term("upright_bonus", "Upright Bonus", "Reward for keeping the pendulum vertical.", 1.0, expression="-abs(pole_angle)"),
    _term("cart_center_penalty", "Cart Center Penalty", "Penalty for moving the cart away from center.", 0.2, expression="-square(cart_position)"),
    _term("angular_velocity_penalty", "Angular Velocity Penalty", "Penalty for fast pendulum rotation.", 0.15, expression="-square(pole_velocity)"),
    _term("action_magnitude_penalty", "Action Magnitude Penalty", "Penalty on strong control pushes.", 0.08, expression="-square(action)"),
]

_INVERTED_DOUBLE_PENDULUM_TEMPLATE = [
    _term("native", "Native Reward", "Original InvertedDoublePendulum reward from Gym.", 1.0, expression="native_reward"),
    _term("upright_bonus", "Upright Bonus", "Reward for keeping the double pendulum upright.", 1.0, expression="pole_angle_1"),
    _term("cart_center_penalty", "Cart Center Penalty", "Penalty for cart displacement.", 0.2, expression="-square(cart_position)"),
    _term("joint_velocity_penalty", "Joint Velocity Penalty", "Penalty for fast joint rotation.", 0.12, expression="-square(pole_velocity_1) - square(pole_velocity_2)"),
    _term("action_magnitude_penalty", "Action Magnitude Penalty", "Penalty on large control pushes.", 0.08, expression="-square(action)"),
]

_CAR_RACING_TEMPLATE = [
    _term("native", "Native Reward", "Original CarRacing reward from Gym.", 1.0, expression="native_reward"),
    _term("throttle_bonus", "Throttle Bonus", "Reward for applying forward throttle.", 0.15, expression="action_1"),
    _term("steering_penalty", "Steering Penalty", "Penalty for aggressive steering.", 0.08, expression="-abs(action_0)"),
    _term("brake_penalty", "Brake Penalty", "Penalty for excessive braking.", 0.1, expression="-action_2"),
    _term("action_change_penalty", "Action Change Penalty", "Penalty for abrupt control changes.", 0.05, expression="-abs(action_0 - prev_action_0) - abs(action_1 - prev_action_1) - abs(action_2 - prev_action_2)"),
]


ENV_TEMPLATE_BY_PREFIX: list[tuple[str, list[dict[str, Any]]]] = [
    ("CartPole-", _CARTPOLE_TEMPLATE),
    ("MountainCarContinuous-", _MOUNTAIN_CAR_CONTINUOUS_TEMPLATE),
    ("MountainCar-", _MOUNTAIN_CAR_TEMPLATE),
    ("Acrobot-", _ACROBOT_TEMPLATE),
    ("Pendulum-", _PENDULUM_TEMPLATE),
    ("LunarLanderContinuous-", _LUNAR_LANDER_TEMPLATE),
    ("LunarLander-", _LUNAR_LANDER_TEMPLATE),
    ("BipedalWalkerHardcore-", _LOCOMOTION_TEMPLATE),
    ("BipedalWalker-", _LOCOMOTION_TEMPLATE),
    ("CarRacing-", _CAR_RACING_TEMPLATE),
    ("HumanoidStandup-", _HUMANOID_STANDUP_TEMPLATE),
    ("Humanoid-", _LOCOMOTION_TEMPLATE),
    ("Ant-", _LOCOMOTION_TEMPLATE),
    ("HalfCheetah-", _LOCOMOTION_TEMPLATE),
    ("Hopper-", _LOCOMOTION_TEMPLATE),
    ("Walker2d-", _LOCOMOTION_TEMPLATE),
    ("Swimmer-", _LOCOMOTION_TEMPLATE),
    ("Reacher-", _REACHER_TEMPLATE),
    ("Pusher-", _PUSHER_TEMPLATE),
    ("InvertedDoublePendulum-", _INVERTED_DOUBLE_PENDULUM_TEMPLATE),
    ("InvertedPendulum-", _INVERTED_PENDULUM_TEMPLATE),
]


def reward_template_for_env(env_name: str) -> list[dict[str, Any]]:
    for prefix, template in ENV_TEMPLATE_BY_PREFIX:
        if env_name.startswith(prefix):
            return deepcopy(template)
    return deepcopy(DEFAULT_REWARD_TEMPLATE)


def _flat_obs(obs) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _flat_action(action) -> np.ndarray:
    if action is None:
        return np.zeros(1, dtype=np.float32)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _action_change(current_action: np.ndarray, previous_action) -> float:
    if previous_action is None:
        return 0.0
    previous = np.asarray(previous_action, dtype=np.float32).reshape(-1)
    if previous.size == 0:
        return 0.0
    if current_action.size == 1:
        return -float(abs(float(current_action[0]) - float(previous[0])))
    if previous.size == 1:
        prev = np.full_like(current_action, float(previous[0]), dtype=np.float32)
    elif previous.size != current_action.size:
        prev = np.resize(previous, current_action.size).astype(np.float32, copy=False)
    else:
        prev = previous.astype(np.float32, copy=False)
    return -float(np.mean(np.abs(current_action - prev)))


def _info_value(info: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in info:
            value = info[key]
            if isinstance(value, (int, float, np.integer, np.floating)):
                return float(value)
            arr = np.asarray(value).reshape(-1)
            if arr.size == 1:
                return float(arr[0])
    return float(default)


def raw_reward_terms_for_env(
    env_name: str,
    obs,
    action,
    previous_action,
    native_reward: float,
    info: dict[str, Any] | None = None,
) -> dict[str, float]:
    info = info or {}
    flat_obs = _flat_obs(obs)
    flat_action = _flat_action(action)

    terms = {"native": float(native_reward)}

    if env_name.startswith("CartPole-"):
        if flat_obs.size >= 4:
            x, x_dot, theta, theta_dot = flat_obs[:4]
            prev_action_value = None if previous_action is None else float(np.asarray(previous_action).reshape(-1)[0])
            terms.update(
                {
                    "survival_bonus": 1.0,
                    "cart_position_penalty": -float(x * x),
                    "cart_velocity_penalty": -float(x_dot * x_dot),
                    "pole_angle_penalty": -float(theta * theta),
                    "pole_velocity_penalty": -float(theta_dot * theta_dot),
                    "action_change_penalty": -1.0 if prev_action_value is not None and int(flat_action[0]) != int(prev_action_value) else 0.0,
                }
            )
        return terms

    if env_name.startswith("MountainCarContinuous-"):
        if flat_obs.size >= 2:
            position, velocity = flat_obs[:2]
            terms.update(
                {
                    "hill_progress_bonus": float(position + 0.5),
                    "speed_bonus": float(abs(velocity)),
                    "throttle_penalty": -float(np.sum(np.square(flat_action))),
                    "action_change_penalty": _action_change(flat_action, previous_action),
                }
            )
        return terms

    if env_name.startswith("MountainCar-"):
        if flat_obs.size >= 2:
            position, velocity = flat_obs[:2]
            prev_action_value = None if previous_action is None else float(np.asarray(previous_action).reshape(-1)[0])
            terms.update(
                {
                    "hill_progress_bonus": float(position + 0.5),
                    "speed_bonus": float(abs(velocity)),
                    "goal_side_bonus": float(max(position, 0.0)),
                    "action_change_penalty": -1.0 if prev_action_value is not None and int(flat_action[0]) != int(prev_action_value) else 0.0,
                }
            )
        return terms

    if env_name.startswith("Acrobot-"):
        if flat_obs.size >= 6:
            cos1, sin1, cos2, sin2, vel1, vel2 = flat_obs[:6]
            theta1 = float(np.arctan2(sin1, cos1))
            theta2 = float(np.arctan2(sin2, cos2))
            tip_height = -np.cos(theta1) - np.cos(theta1 + theta2)
            prev_action_value = None if previous_action is None else float(np.asarray(previous_action).reshape(-1)[0])
            terms.update(
                {
                    "tip_height_bonus": float(tip_height),
                    "swing_momentum_bonus": float(abs(vel1) + abs(vel2)),
                    "joint_velocity_penalty": -float(vel1 * vel1 + vel2 * vel2),
                    "action_change_penalty": -1.0 if prev_action_value is not None and int(flat_action[0]) != int(prev_action_value) else 0.0,
                }
            )
        return terms

    if env_name.startswith("Pendulum-"):
        if flat_obs.size >= 3:
            cos_theta, _, theta_dot = flat_obs[:3]
            terms.update(
                {
                    "upright_bonus": float(cos_theta),
                    "angular_velocity_penalty": -float(theta_dot * theta_dot),
                    "torque_penalty": -float(np.sum(np.square(flat_action))),
                    "action_change_penalty": _action_change(flat_action, previous_action),
                }
            )
        return terms

    if env_name.startswith("LunarLander"):
        if flat_obs.size >= 8:
            x, _, x_vel, y_vel, angle, _, left_leg, right_leg = flat_obs[:8]
            terms.update(
                {
                    "centering_penalty": -float(x * x),
                    "landing_speed_penalty": -float(x_vel * x_vel + y_vel * y_vel),
                    "angle_penalty": -float(angle * angle),
                    "leg_contact_bonus": float(left_leg + right_leg),
                    "fuel_penalty": -float(np.sum(np.square(flat_action))),
                }
            )
        return terms

    if env_name.startswith("CarRacing-"):
        steer = float(flat_action[0]) if flat_action.size > 0 else 0.0
        gas = float(flat_action[1]) if flat_action.size > 1 else 0.0
        brake = float(flat_action[2]) if flat_action.size > 2 else 0.0
        terms.update(
            {
                "throttle_bonus": gas,
                "steering_penalty": -abs(steer),
                "brake_penalty": -brake,
                "action_change_penalty": _action_change(flat_action, previous_action),
            }
        )
        return terms

    if env_name.startswith("Reacher-"):
        distance_penalty = _info_value(info, "reward_dist", default=0.0)
        if distance_penalty == 0.0 and flat_obs.size >= 2:
            distance_penalty = -float(np.linalg.norm(flat_obs[-2:]))
        terms.update(
            {
                "target_proximity_bonus": -distance_penalty,
                "distance_penalty": float(distance_penalty),
                "control_penalty": _info_value(info, "reward_ctrl", default=-float(np.sum(np.square(flat_action)))),
                "action_change_penalty": _action_change(flat_action, previous_action),
            }
        )
        return terms

    if env_name.startswith("Pusher-"):
        object_to_goal = -_info_value(info, "reward_dist", default=0.0)
        hand_to_object = -_info_value(info, "reward_near", default=0.0)
        terms.update(
            {
                "object_to_goal_bonus": float(object_to_goal),
                "hand_to_object_bonus": float(hand_to_object),
                "control_penalty": _info_value(info, "reward_ctrl", default=-float(np.sum(np.square(flat_action)))),
                "action_change_penalty": _action_change(flat_action, previous_action),
            }
        )
        return terms

    if env_name.startswith("HumanoidStandup-"):
        balance_obs = flat_obs[:5] if flat_obs.size >= 5 else flat_obs
        terms.update(
            {
                "standup_bonus": _info_value(info, "uph_cost", "reward_linup", default=float(flat_obs[0]) if flat_obs.size else 0.0),
                "control_penalty": _info_value(info, "reward_quadctrl", "reward_ctrl", default=-float(np.sum(np.square(flat_action)))),
                "contact_penalty": _info_value(info, "reward_impact", "reward_contact", default=0.0),
                "balance_penalty": -float(np.mean(np.square(balance_obs))) if balance_obs.size else 0.0,
                "action_change_penalty": _action_change(flat_action, previous_action),
            }
        )
        return terms

    if env_name.startswith(("Humanoid-", "Ant-", "HalfCheetah-", "Hopper-", "Walker2d-", "Swimmer-", "BipedalWalker")):
        posture_obs = flat_obs[:6] if flat_obs.size >= 6 else flat_obs
        terms.update(
            {
                "forward_bonus": _info_value(
                    info,
                    "reward_forward",
                    "reward_run",
                    "reward_linvel",
                    "x_velocity",
                    default=float(flat_obs[0]) if flat_obs.size else 0.0,
                ),
                "healthy_bonus": _info_value(
                    info,
                    "reward_survive",
                    "reward_alive",
                    "healthy_reward",
                    default=1.0,
                ),
                "control_penalty": _info_value(
                    info,
                    "reward_ctrl",
                    "reward_quadctrl",
                    default=-float(np.sum(np.square(flat_action))),
                ),
                "contact_penalty": _info_value(info, "reward_contact", "reward_impact", default=0.0),
                "stability_penalty": -float(np.mean(np.square(posture_obs))) if posture_obs.size else 0.0,
                "action_change_penalty": _action_change(flat_action, previous_action),
            }
        )
        return terms

    if env_name.startswith("InvertedDoublePendulum-"):
        if flat_obs.size >= 6:
            cart_x = flat_obs[0]
            joint_vel = flat_obs[-3:]
            upright_signal = float(flat_obs[1] if flat_obs.size > 1 else 0.0)
            terms.update(
                {
                    "upright_bonus": upright_signal,
                    "cart_center_penalty": -float(cart_x * cart_x),
                    "joint_velocity_penalty": -float(np.mean(np.square(joint_vel))),
                    "action_magnitude_penalty": -float(np.sum(np.square(flat_action))),
                }
            )
        return terms

    if env_name.startswith("InvertedPendulum-"):
        if flat_obs.size >= 4:
            cart_x = flat_obs[0]
            pole_angle = flat_obs[1]
            pole_vel = flat_obs[-1]
            terms.update(
                {
                    "upright_bonus": -abs(float(pole_angle)),
                    "cart_center_penalty": -float(cart_x * cart_x),
                    "angular_velocity_penalty": -float(pole_vel * pole_vel),
                    "action_magnitude_penalty": -float(np.sum(np.square(flat_action))),
                }
            )
        return terms

    terms.update(
        {
            "action_magnitude_penalty": -float(np.sum(np.square(flat_action))),
            "action_change_penalty": _action_change(flat_action, previous_action),
        }
    )
    return terms
