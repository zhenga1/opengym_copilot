from __future__ import annotations

from typing import Any
from run_logging import (
    LOGS_DIR,
    configure_backend_logging,
    log_reward_spec_snapshot,
    log_run_event,
)

def _env_prefix(env_name: str | None) -> str:
    return (env_name or "").split("/", 1)[-1].split("-", 1)[0].lower()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rule(metric: str, op: str, threshold: float, weight: float = 1.0) -> dict[str, Any]:
    return {
        "metric": metric,
        "op": op,
        "threshold": float(threshold),
        "weight": float(weight),
    }


def _metric_candidates_rule(metrics: list[str], op: str, threshold: float, weight: float = 1.0) -> dict[str, Any]:
    return {
        "metrics": [str(metric).strip() for metric in metrics if str(metric).strip()],
        "op": op,
        "threshold": float(threshold),
        "weight": float(weight),
    }


# Get the behavior Tags 
BEHAVIOR_TAG_LIBRARY: dict[str, list[dict[str, Any]]] = {
    "generic": [
        {
            "key": "sustained_balance",
            "title": "Sustained Balance",
            "description": "Maintain the system in a controlled stable regime for long episodes.",
            "polarity": "desired",
            "rules": [_rule("episode_length", ">=", 150.0, 1.2)],
            "tags": ["stability", "survival"],
        },
        {
            "key": "smooth_control",
            "title": "Smooth Control",
            "description": "Use low-jerk, low-oscillation actuation rather than noisy or impulsive control.",
            "polarity": "desired",
            "rules": [_rule("action_0_abs_mean", "<=", 0.8), _rule("action_0_sign_change_fraction", "<=", 0.35)],
            "tags": ["control", "smoothness"],
        },
        {
            "key": "jerky_control",
            "title": "Jerky Control",
            "description": "High-frequency, inconsistent, or aggressively reversing control.",
            "polarity": "avoid",
            "rules": [_rule("action_0_sign_change_fraction", ">=", 0.55), _rule("action_0_std", ">=", 0.65)],
            "mode": "any",
            "tags": ["control", "instability"],
        },
        {
            "key": "low_motion",
            "title": "Low Motion",
            "description": "Barely move, often indicating under-exploration or frozen policy behavior.",
            "polarity": "avoid",
            "rules": [_rule("action_0_std", "<=", 0.05)],
            "mode": "any",
            "tags": ["exploration", "stagnation"],
        },
        {
            "key": "high_motion",
            "title": "High Motion",
            "description": "Move energetically with substantial state variation.",
            "polarity": "desired",
            "rules": [_rule("action_0_std", ">=", 0.25)],
            "mode": "any",
            "tags": ["exploration", "activity"],
        },
    ],
    "cartpole": [
        {
            "key": "alternating_motion",
            "title": "Alternating Motion",
            "description": "Move back and forth with repeated reversals rather than staying on one side.",
            "polarity": "desired",
            "rules": [_rule("cart_position_sign_change_count", ">=", 2.0), _rule("pole_angle_sign_change_count", ">=", 2.0)],
            "tags": ["cartpole", "oscillation", "alternation"],
        },
        {
            "key": "single_direction_motion",
            "title": "Single-Direction Motion",
            "description": "Drift or commit to one direction with little meaningful reversal.",
            "polarity": "avoid",
            "rules": [_rule("cart_velocity_positive_fraction", ">=", 0.9), _rule("cart_velocity_negative_fraction", ">=", 0.9)],
            "mode": "any",
            "tags": ["cartpole", "drift", "bias"],
        },
        {
            "key": "bounded_cart_motion",
            "title": "Bounded Cart Motion",
            "description": "Move left and right while keeping the cart reasonably centered and recoverable.",
            "polarity": "desired",
            "rules": [_rule("cart_position_abs_mean", "<=", 0.9), _rule("cart_position_range", "<=", 3.5)],
            "tags": ["cartpole", "cart"],
        },
        {
            "key": "runaway_cart",
            "title": "Runaway Cart",
            "description": "Let the cart run too far from center or build unrecoverable lateral drift.",
            "polarity": "avoid",
            "rules": [_rule("cart_position_abs_mean", ">=", 1.4), _rule("cart_position_range", ">=", 4.0)],
            "mode": "any",
            "tags": ["cartpole", "cart", "failure"],
        },
        {
            "key": "one_sided_cart_motion",
            "title": "One-Sided Cart Motion",
            "description": "Spend most time on one side instead of alternating coverage.",
            "polarity": "avoid",
            "rules": [_rule("cart_position_left_time_fraction", ">=", 0.8), _rule("cart_position_right_time_fraction", ">=", 0.8)],
            "mode": "any",
            "tags": ["cartpole", "bias"],
        },
        {
            "key": "balanced_side_occupancy",
            "title": "Balanced Side Occupancy",
            "description": "Occupy left and right sides with roughly similar fractions.",
            "polarity": "desired",
            "rules": [_rule("cart_side_balance_score", ">=", 0.6)],
            "tags": ["cartpole", "balance", "alternation"],
        },
        {
            "key": "upright_posture",
            "title": "Upright Posture",
            "description": "Remain upright with modest pole-angle and angular-velocity excursions.",
            "polarity": "desired",
            "rules": [_rule("pole_angle_abs_mean", "<=", 0.45), _rule("pole_velocity_abs_mean", "<=", 1.2)],
            "tags": ["cartpole", "posture", "stability"],
        },
        {
            "key": "excessive_tilt",
            "title": "Excessive Tilt",
            "description": "Spend too much time strongly tilted or falling rapidly.",
            "polarity": "avoid",
            "rules": [_rule("pole_angle_abs_mean", ">=", 0.35), _rule("pole_velocity_abs_mean", ">=", 1.8)],
            "mode": "any",
            "tags": ["cartpole", "posture", "instability"],
        },
        {
            "key": "high_reversal_frequency",
            "title": "High Reversal Frequency",
            "description": "Reverse cart direction and pole direction frequently.",
            "polarity": "desired",
            "rules": [_rule("cart_velocity_sign_change_count", ">=", 3.0), _rule("pole_angle_sign_change_count", ">=", 3.0)],
            "mode": "any",
            "tags": ["cartpole", "oscillation", "reversal"],
        },
        {
            "key": "low_reversal_frequency",
            "title": "Low Reversal Frequency",
            "description": "Rarely reverse direction or sign, indicating sticky or one-sided behavior.",
            "polarity": "avoid",
            "rules": [_rule("cart_velocity_sign_change_count", "<=", 1.0), _rule("pole_angle_sign_change_count", "<=", 1.0)],
            "mode": "any",
            "tags": ["cartpole", "oscillation", "failure"],
        },
        {
            "key": "pole_recovery",
            "title": "Pole Recovery",
            "description": "Recover pole angle direction repeatedly instead of falling to one side.",
            "polarity": "desired",
            "rules": [_rule("pole_angle_sign_change_count", ">=", 2.0), _rule("pole_angle_abs_mean", "<=", 0.35)],
            "tags": ["cartpole", "pole", "recovery"],
        },
        {
            "key": "missing_return_phase",
            "title": "Missing Return Phase",
            "description": "Fail to complete the return half of a desired left-right oscillation.",
            "polarity": "avoid",
            "rules": [_rule("pole_angle_sign_change_count", "<=", 1.0), _rule("cart_side_balance_score", "<=", 0.35)],
            "tags": ["cartpole", "alternation", "failure"],
        },
        {
            "key": "periodic_sway_tracking",
            "title": "Periodic Sway Tracking",
            "description": "Track a periodic target sway reference rather than merely surviving.",
            "polarity": "desired",
            "rules": [_rule("tracking_error_abs_mean", "<=", 0.18)],
            "tags": ["cartpole", "tracking", "oscillation"],
        },
    ],
    "mountaincar": [
        {
            "key": "alternating_motion",
            "title": "Alternating Motion",
            "description": "Reverse direction repeatedly to build momentum instead of committing to one slope.",
            "polarity": "desired",
            "rules": [_rule("velocity_reversal_count", ">=", 4.0)],
            "tags": ["mountaincar", "oscillation", "momentum"],
        },
        {
            "key": "low_reversal_frequency",
            "title": "Low Reversal Frequency",
            "description": "Rarely reverse velocity, which usually prevents useful momentum building.",
            "polarity": "avoid",
            "rules": [_rule("velocity_reversal_count", "<=", 1.0)],
            "tags": ["mountaincar", "oscillation", "failure"],
        },
        {
            "key": "hill_climbing_progress",
            "title": "Hill Climbing Progress",
            "description": "Spend more time on the goal-side hill and improve position over time.",
            "polarity": "desired",
            "rules": [_rule("hill_progress_fraction", ">=", 0.35), _rule("position_delta", ">=", 0.15)],
            "tags": ["mountaincar", "progress"],
        },
        {
            "key": "momentum_building",
            "title": "Momentum Building",
            "description": "Reverse velocity repeatedly to build momentum for the climb.",
            "polarity": "desired",
            "rules": [_rule("velocity_reversal_count", ">=", 4.0)],
            "tags": ["mountaincar", "momentum"],
        },
        {
            "key": "valley_stuck",
            "title": "Valley Stuck",
            "description": "Remain near the valley with little progress or momentum.",
            "polarity": "avoid",
            "rules": [_rule("hill_progress_fraction", "<=", 0.1), _rule("velocity_reversal_count", "<=", 1.0)],
            "tags": ["mountaincar", "failure"],
        },
    ],
    "pendulum": [
        {
            "key": "alternating_motion",
            "title": "Alternating Motion",
            "description": "Sweep the pendulum through repeated angle reversals rather than staying stuck on one side.",
            "polarity": "desired",
            "rules": [_rule("pendulum_angle_sign_change_count", ">=", 2.0)],
            "tags": ["pendulum", "oscillation", "alternation"],
        },
        {
            "key": "upright_posture",
            "title": "Upright Posture",
            "description": "Keep the pendulum near upright with modest angular velocity.",
            "polarity": "desired",
            "rules": [_rule("pendulum_angle_abs_mean", "<=", 0.45), _rule("angular_velocity_abs_mean", "<=", 1.2)],
            "tags": ["pendulum", "posture", "stability"],
        },
        {
            "key": "excessive_tilt",
            "title": "Excessive Tilt",
            "description": "Spend too much time far from upright or rotating too aggressively.",
            "polarity": "avoid",
            "rules": [_rule("pendulum_angle_abs_mean", ">=", 0.35), _rule("angular_velocity_abs_mean", ">=", 1.8)],
            "mode": "any",
            "tags": ["pendulum", "posture", "instability"],
        },
        {
            "key": "upright_stabilization",
            "title": "Upright Stabilization",
            "description": "Keep the pendulum near upright with limited angular velocity.",
            "polarity": "desired",
            "rules": [_rule("pendulum_angle_abs_mean", "<=", 0.5), _rule("angular_velocity_abs_mean", "<=", 2.2)],
            "tags": ["pendulum", "upright"],
        },
        {
            "key": "swing_up_motion",
            "title": "Swing-Up Motion",
            "description": "Generate enough angular movement to swing toward the upright regime.",
            "polarity": "desired",
            "rules": [_rule("pendulum_angle_range", ">=", 2.5), _rule("angular_velocity_std", ">=", 0.8)],
            "tags": ["pendulum", "swing_up"],
        },
        {
            "key": "excessive_spin",
            "title": "Excessive Spin",
            "description": "Rotate too aggressively instead of stabilizing or purposeful swing-up.",
            "polarity": "avoid",
            "rules": [_rule("angular_velocity_abs_mean", ">=", 4.0), _rule("pendulum_angle_sign_change_count", ">=", 6.0)],
            "tags": ["pendulum", "instability"],
        },
    ],
    "acrobot": [
        {
            "key": "alternating_motion",
            "title": "Alternating Motion",
            "description": "Use repeated joint-angle reversals to swing through the trajectory instead of freezing one side.",
            "polarity": "desired",
            "rules": [_rule("joint1_angle_sign_change_count", ">=", 2.0), _rule("joint2_angle_sign_change_count", ">=", 2.0)],
            "tags": ["acrobot", "oscillation", "alternation"],
        },
        {
            "key": "double_link_swing_up",
            "title": "Double-Link Swing-Up",
            "description": "Use both links dynamically to build energy and swing upward.",
            "polarity": "desired",
            "rules": [_rule("joint1_angle_range", ">=", 1.0), _rule("joint2_angle_range", ">=", 1.0)],
            "tags": ["acrobot", "swing_up"],
        },
        {
            "key": "low_joint_excitation",
            "title": "Low Joint Excitation",
            "description": "Fail to excite one or both joints enough to build useful energy.",
            "polarity": "avoid",
            "rules": [_rule("joint1_angle_std", "<=", 0.2), _rule("joint2_angle_std", "<=", 0.2)],
            "mode": "all",
            "tags": ["acrobot", "stagnation"],
        },
    ],
    "lunarlander": [
        {
            "key": "upright_posture",
            "title": "Upright Posture",
            "description": "Maintain a mostly upright lander attitude with controlled angular velocity.",
            "polarity": "desired",
            "rules": [_rule("angle_abs_mean", "<=", 0.45), _rule("angular_velocity_abs_mean", "<=", 1.2)],
            "tags": ["lander", "posture", "stability"],
        },
        {
            "key": "excessive_tilt",
            "title": "Excessive Tilt",
            "description": "Remain overly tilted or rotating too aggressively during descent.",
            "polarity": "avoid",
            "rules": [_rule("angle_abs_mean", ">=", 0.35), _rule("angular_velocity_abs_mean", ">=", 1.8)],
            "mode": "any",
            "tags": ["lander", "posture", "instability"],
        },
        {
            "key": "symmetric_left_right_usage",
            "title": "Symmetric Left/Right Usage",
            "description": "Use both landing legs in a balanced way during touchdown.",
            "polarity": "desired",
            "rules": [_rule("left_leg_contact_true_fraction", ">=", 0.05), _rule("right_leg_contact_true_fraction", ">=", 0.05)],
            "tags": ["lander", "symmetry", "contact"],
        },
        {
            "key": "stable_attitude",
            "title": "Stable Attitude",
            "description": "Maintain a controlled low-tilt orientation during descent.",
            "polarity": "desired",
            "rules": [_rule("tilted_fraction", "<=", 0.35), _rule("angular_velocity_abs_mean", "<=", 1.5)],
            "tags": ["lander", "attitude"],
        },
        {
            "key": "gentle_descent",
            "title": "Gentle Descent",
            "description": "Descend with limited vertical speed and controlled rotation.",
            "polarity": "desired",
            "rules": [_rule("y_velocity_abs_mean", "<=", 0.8), _rule("angular_velocity_abs_mean", "<=", 1.5)],
            "tags": ["lander", "descent"],
        },
        {
            "key": "leg_contact_balance",
            "title": "Leg Contact Balance",
            "description": "Use left and right leg contacts in a balanced way during landing.",
            "polarity": "desired",
            "rules": [_rule("left_leg_contact_true_fraction", ">=", 0.05), _rule("right_leg_contact_true_fraction", ">=", 0.05)],
            "tags": ["lander", "landing", "contact"],
        },
        {
            "key": "persistent_lander_tilt",
            "title": "Persistent Lander Tilt",
            "description": "Remain significantly tilted for much of descent.",
            "polarity": "avoid",
            "rules": [_rule("tilted_fraction", ">=", 0.6)],
            "tags": ["lander", "failure", "tilt"],
        },
    ],
    "reacher": [
        {
            "key": "target_tracking",
            "title": "Target Tracking",
            "description": "Stay close to the target rather than merely moving the arm around.",
            "polarity": "desired",
            "rules": [_rule("target_distance_mean", "<=", 0.25)],
            "tags": ["reacher", "tracking", "target"],
        },
        {
            "key": "target_approach",
            "title": "Target Approach",
            "description": "Reduce the fingertip-to-target distance over the episode.",
            "polarity": "desired",
            "rules": [_rule("target_distance_delta", "<=", -0.05)],
            "tags": ["reacher", "tracking", "progress"],
        },
        {
            "key": "target_proximity",
            "title": "Target Proximity",
            "description": "Keep the fingertip close to the target.",
            "polarity": "desired",
            "rules": [_rule("target_distance_mean", "<=", 0.15)],
            "tags": ["reacher", "target"],
        },
        {
            "key": "consistent_target_approach",
            "title": "Consistent Target Approach",
            "description": "Reduce target distance over the course of the episode.",
            "polarity": "desired",
            "rules": [_rule("target_distance_delta", "<=", -0.05)],
            "tags": ["reacher", "progress"],
        },
        {
            "key": "stalled_target_approach",
            "title": "Stalled Target Approach",
            "description": "Make little progress toward the target.",
            "polarity": "avoid",
            "rules": [_rule("target_distance_delta", ">=", -0.01)],
            "tags": ["reacher", "failure"],
        },
    ],
    "locomotion": [
        {
            "key": "single_direction_motion",
            "title": "Single-Direction Motion",
            "description": "Commit strongly to one travel direction with little meaningful reversal.",
            "polarity": "avoid",
            "rules": [_rule("x_velocity_positive_fraction", ">=", 0.9), _rule("x_velocity_negative_fraction", ">=", 0.9)],
            "mode": "any",
            "tags": ["locomotion", "drift", "bias"],
        },
        {
            "key": "symmetric_left_right_usage",
            "title": "Symmetric Left/Right Usage",
            "description": "Use left and right limbs with roughly similar intensity.",
            "polarity": "desired",
            "rules": [
                _rule("left_right_thigh_angle_balance_score", ">=", 0.5),
                _rule("left_right_leg_angle_balance_score", ">=", 0.5),
                _rule("left_right_foot_angle_balance_score", ">=", 0.5),
            ],
            "mode": "any",
            "tags": ["locomotion", "symmetry", "coverage"],
        },
        {
            "key": "forward_progress",
            "title": "Forward Progress",
            "description": "Produce sustained positive forward velocity.",
            "polarity": "desired",
            "rules": [_rule("x_velocity_mean", ">=", 0.5), _rule("x_velocity_positive_fraction", ">=", 0.6)],
            "tags": ["locomotion", "forward"],
        },
        {
            "key": "backward_drift",
            "title": "Backward Drift",
            "description": "Spend much of the trajectory moving backwards.",
            "polarity": "avoid",
            "rules": [_rule("x_velocity_mean", "<=", -0.2), _rule("x_velocity_negative_fraction", ">=", 0.65)],
            "tags": ["locomotion", "backward"],
        },
        {
            "key": "lateral_drift",
            "title": "Lateral Drift",
            "description": "Move sideways instead of maintaining a stable heading corridor.",
            "polarity": "avoid",
            "rules": [_rule("y_velocity_abs_mean", ">=", 0.3)],
            "tags": ["locomotion", "heading"],
        },
        {
            "key": "upright_posture",
            "title": "Upright Posture",
            "description": "Remain upright with modest pitch and roll excursions.",
            "polarity": "desired",
            "rules": [_rule("torso_pitch_abs_mean", "<=", 0.45), _rule("roll_velocity_abs_mean", "<=", 1.2)],
            "mode": "any",
            "tags": ["locomotion", "posture", "stability"],
        },
        {
            "key": "excessive_tilt",
            "title": "Excessive Tilt",
            "description": "Spend too much time strongly pitched or rolled while moving.",
            "polarity": "avoid",
            "rules": [_rule("torso_pitch_abs_mean", ">=", 0.35), _rule("pitch_velocity_abs_mean", ">=", 1.8)],
            "mode": "any",
            "tags": ["locomotion", "posture", "instability"],
        },
        {
            "key": "high_reversal_frequency",
            "title": "High Reversal Frequency",
            "description": "Reverse the forward velocity sign often, indicating oscillatory or unstable locomotion.",
            "polarity": "desired",
            "rules": [_rule("x_velocity_sign_change_count", ">=", 3.0)],
            "tags": ["locomotion", "oscillation", "reversal"],
        },
        {
            "key": "low_reversal_frequency",
            "title": "Low Reversal Frequency",
            "description": "Rarely reverse direction or sign, indicating sticky or one-sided locomotion.",
            "polarity": "avoid",
            "rules": [_rule("x_velocity_sign_change_count", "<=", 1.0)],
            "tags": ["locomotion", "oscillation", "failure"],
        },
        {
            "key": "forward_locomotion",
            "title": "Forward Locomotion",
            "description": "Move forward with sustained positive x velocity.",
            "polarity": "desired",
            "rules": [_rule("x_velocity_mean", ">=", 0.8), _rule("x_velocity_positive_fraction", ">=", 0.7)],
            "tags": ["locomotion", "forward"],
        },
        {
            "key": "stable_heading",
            "title": "Stable Heading",
            "description": "Limit unwanted yaw drift while moving.",
            "polarity": "desired",
            "rules": [_rule("yaw_velocity_abs_mean", "<=", 1.0)],
            "tags": ["locomotion", "heading"],
        },
        {
            "key": "gait_symmetry",
            "title": "Gait Symmetry",
            "description": "Use left and right limbs with roughly balanced motion.",
            "polarity": "desired",
            "rules": [
                _rule("left_right_thigh_angle_balance_score", ">=", 0.5),
                _rule("left_right_leg_angle_balance_score", ">=", 0.5),
                _rule("left_right_foot_angle_balance_score", ">=", 0.5),
            ],
            "mode": "any",
            "tags": ["locomotion", "symmetry", "gait"],
        },
        {
            "key": "vertical_bounce_control",
            "title": "Vertical Bounce Control",
            "description": "Avoid excessive up-down bouncing while moving.",
            "polarity": "desired",
            "rules": [_rule("z_velocity_std", "<=", 1.0)],
            "tags": ["locomotion", "vertical_stability"],
        },
        {
            "key": "excessive_pitch_motion",
            "title": "Excessive Pitch Motion",
            "description": "Pitch too much while attempting locomotion.",
            "polarity": "avoid",
            "rules": [_rule("pitch_velocity_abs_mean", ">=", 1.8), _rule("torso_pitch_abs_mean", ">=", 0.5)],
            "mode": "any",
            "tags": ["locomotion", "instability"],
        },
    ],
}


ENV_GROUPS = {
    "cartpole": {"cartpole", "invertedpendulum", "inverteddoublependulum"},
    "mountaincar": {"mountaincar", "mountaincarcontinuous"},
    "pendulum": {"pendulum"},
    "acrobot": {"acrobot"},
    "lunarlander": {"lunarlander"},
    "reacher": {"reacher", "pusher"},
    "locomotion": {"halfcheetah", "hopper", "walker2d", "ant", "humanoid", "swimmer"},
}
logger = configure_backend_logging()

def available_behavior_tags_for_env(env_name: str | None) -> list[dict[str, Any]]:
    """
    Return a list of dictionaries representing behavior tags for given environment. 
    Each dictionary contains:
    - key: unique identifier for the tag
    - title: human-readable title for the tag
    - description: detailed explanation of the tag
    - polarity: "desired" or "avoid" indicating whether the tag represents a positive or
      negative behavior
    - rules: a list of dictionaries specifying the metric-based rules for scoring this tag
    - tags: a list of additional keywords associated with the tag for categorization and searchability
    """
    prefix = _env_prefix(env_name)
    logger.info(f"Requested behavior tags for environment: {env_name} (prefix: {prefix})")
    tags = list(BEHAVIOR_TAG_LIBRARY["generic"])
    for group_name, env_prefixes in ENV_GROUPS.items():
        # Check if the environment is in the group based on prefix matching
        if prefix in env_prefixes:
            tags.extend(BEHAVIOR_TAG_LIBRARY[group_name])
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tag in tags:
        key = str(tag.get("key") or "").strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(tag)
    return deduped


def _score_rule(metric_value: float | None, rule: dict[str, Any]) -> float:
    if metric_value is None:
        return 0.0
    threshold = _safe_float(rule.get("threshold"))
    op = str(rule.get("op") or ">=")
    if op == ">=":
        if threshold == 0:
            return 1.0 if metric_value >= 0 else 0.0
        return max(0.0, min(1.0, metric_value / threshold))
    if op == "<=":
        if metric_value <= threshold:
            return 1.0
        divisor = max(abs(threshold), 1e-6)
        overflow = (metric_value - threshold) / divisor
        return max(0.0, 1.0 - overflow)
    if op == ">":
        return 1.0 if metric_value > threshold else 0.0
    if op == "<":
        return 1.0 if metric_value < threshold else 0.0
    return 0.0


def _score_rule_candidates(metrics: dict[str, Any], rule: dict[str, Any]) -> tuple[float, dict[str, float]]:
    metric_names = [str(rule.get("metric") or "").strip()] if str(rule.get("metric") or "").strip() else []
    metric_names.extend(str(name).strip() for name in (rule.get("metrics") or []) if str(name).strip())

    scored: list[tuple[str, float, float]] = []
    evidence: dict[str, float] = {}
    for metric_name in metric_names:
        metric_value = metrics.get(metric_name)
        if metric_value is None:
            continue
        score = _score_rule(metric_value, rule)
        scored.append((metric_name, score, _safe_float(metric_value)))
        evidence[metric_name] = _safe_float(metric_value)

    if not scored:
        return 0.0, {}
    best_metric_name, best_score, best_value = max(scored, key=lambda item: item[1])
    return best_score, {best_metric_name: best_value}


def build_behavior_tag_report(behavior_report: dict[str, Any] | None, env_name: str | None) -> dict[str, Any]:
    metrics = dict((behavior_report or {}).get("metrics") or {})
    tag_specs = available_behavior_tags_for_env(env_name)

    # INITIALIZE AN EMPTY LIST TO HOLD SCORED TAGS
    scored_tags: list[dict[str, Any]] = []

    for spec in tag_specs:
        # Extract list of rules from the spec. 
        # This should be defined in the default list
        rules = list(spec.get("rules") or [])
        if not rules:
            continue
        weighted_scores: list[tuple[float, float]] = []
        evidence: dict[str, float] = {}
        for rule in rules:
            score, rule_evidence = _score_rule_candidates(metrics, rule)
            weight = _safe_float(rule.get("weight"), 1.0)
            weighted_scores.append((score, weight))
            evidence.update(rule_evidence)
        if not weighted_scores:
            continue
        mode = str(spec.get("mode") or "all")
        if mode == "any":
            final_score = max(score for score, _ in weighted_scores)
        else:
            total_weight = sum(weight for _, weight in weighted_scores) or 1.0
            final_score = sum(score * weight for score, weight in weighted_scores) / total_weight
        scored_tags.append({
            "key": spec["key"],
            "title": spec.get("title", spec["key"]),
            "description": spec.get("description", ""),
            "polarity": spec.get("polarity", "desired"),
            "score": round(final_score, 3),
            "evidence": evidence,
            "tags": list(spec.get("tags") or []),
        })

    supported = [tag for tag in scored_tags if tag["score"] >= 0.6]
    tentative = [tag for tag in scored_tags if 0.35 <= tag["score"] < 0.6]
    supported.sort(key=lambda tag: (-tag["score"], tag["key"]))
    tentative.sort(key=lambda tag: (-tag["score"], tag["key"]))
    return {
        "summary": {
            "env_name": env_name,
            "supported_count": len(supported),
            "tentative_count": len(tentative),
            "catalog_size": len(tag_specs),
        },
        "supported_tags": supported[:20],
        "tentative_tags": tentative[:20],
        "all_tag_scores": scored_tags,
        "available_tags": [
            {
                "key": spec["key"],
                "title": spec.get("title", spec["key"]),
                "description": spec.get("description", ""),
                "polarity": spec.get("polarity", "desired"),
                "tags": list(spec.get("tags") or []),
            }
            for spec in tag_specs
        ],
    }


def heuristic_behavior_tag_plan(goal: str, env_name: str | None) -> dict[str, Any]:
    lower_goal = str(goal or "").strip().lower()
    desired_tags: list[dict[str, Any]] = []
    avoid_tags: list[dict[str, Any]] = []
    constraints: list[str] = []

    def add_desired(key: str, weight: float, reason: str) -> None:
        desired_tags.append({"key": key, "weight": round(max(0.0, min(weight, 1.0)), 2), "reason": reason})

    def add_avoid(key: str, weight: float, reason: str) -> None:
        avoid_tags.append({"key": key, "weight": round(max(0.0, min(weight, 1.0)), 2), "reason": reason})

    if any(token in lower_goal for token in ("left and right", "left-right", "back and forth", "oscillat", "sway")):
        add_desired("alternating_motion", 0.95, "The goal explicitly asks for repeated left-right motion.")
        add_desired("balanced_side_occupancy", 0.85, "Alternating motion should use both sides instead of one.")
        add_avoid("one_sided_cart_motion", 0.95, "One-sided drift directly contradicts the requested alternating behavior.")
        add_avoid("missing_return_phase", 0.9, "The return half-cycle is required for true back-and-forth behavior.")
    if any(token in lower_goal for token in ("balance", "upright", "stable")):
        add_desired("sustained_balance", 0.85, "The goal requires stable survival or balance.")
        add_desired("upright_posture", 0.8, "Upright posture typically supports balancing tasks.")
        add_avoid("excessive_tilt", 0.8, "Large tilt undermines stable balance.")
    if any(token in lower_goal for token in ("smooth", "graceful", "gentle")):
        add_desired("smooth_control", 0.7, "The goal values smooth actuation.")
        add_avoid("jerky_control", 0.75, "Jerky control conflicts with smooth execution.")
    if any(token in lower_goal for token in ("forward", "run", "walk", "hop")):
        add_desired("forward_progress", 0.95, "The goal requires positive forward movement.")
        add_desired("forward_locomotion", 0.9, "Locomotion tasks should exhibit sustained positive x velocity.")
        add_avoid("backward_drift", 0.8, "Backward drift conflicts with forward movement.")
    if any(token in lower_goal for token in ("target", "reach", "touch")):
        add_desired("target_tracking", 0.9, "The goal requires staying close to a target.")
        add_desired("target_approach", 0.8, "The agent should reduce target distance over time.")
        add_avoid("stalled_target_approach", 0.75, "Stalling away from the target contradicts the task.")
    if any(token in lower_goal for token in ("land", "landing")):
        add_desired("gentle_descent", 0.85, "Landing tasks should reduce vertical speed before touchdown.")
        add_desired("stable_attitude", 0.8, "Controlled attitude is important for stable landing.")
        add_avoid("persistent_lander_tilt", 0.8, "Persistent tilt is unsafe during landing.")
    if not desired_tags:
        add_desired("sustained_balance", 0.55, "Fallback default: prefer long stable episodes.")
        add_desired("smooth_control", 0.45, "Fallback default: prefer smoother control over noisy actuation.")
        add_avoid("jerky_control", 0.45, "Fallback default: reduce noisy control.")

    constraints.append("Favor behavior tags with strong support in the environment's deterministic metric library.")
    constraints.append("Prefer a small set of high-importance desired tags plus a few explicit anti-goals.")

    return {
        "goal": str(goal or "").strip(),
        "desired_tags": desired_tags,
        "avoid_tags": avoid_tags,
        "constraints": constraints,
        "rationale": "Heuristic fallback behavior-plan derived from goal keywords.",
        "provider": "heuristic",
        "model": "",
    }


def normalize_behavior_tag_plan(plan: dict[str, Any] | None, env_name: str | None, goal: str | None = None) -> dict[str, Any]:
    catalog = {spec["key"]: spec for spec in available_behavior_tags_for_env(env_name)}

    def _normalize_items(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items or []:
            key = str(item.get("key") or "").strip()
            if not key or key not in catalog or key in seen:
                continue
            seen.add(key)
            normalized.append({
                "key": key,
                "weight": round(max(0.0, min(_safe_float(item.get("weight"), 0.5), 1.0)), 2),
                "reason": str(item.get("reason") or catalog[key].get("description") or "").strip(),
            })
        return normalized

    current = dict(plan or {})
    current["goal"] = str(current.get("goal") or goal or "").strip()
    current["desired_tags"] = _normalize_items(current.get("desired_tags"))
    current["avoid_tags"] = _normalize_items(current.get("avoid_tags"))
    current["constraints"] = [str(item).strip() for item in (current.get("constraints") or []) if str(item).strip()]
    current["rationale"] = str(current.get("rationale") or "").strip()
    return current
