from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any

import numpy as np

from train_backend_reward_tuning.reward_shaping import reward_expression_variable_specs


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_prefix(env_name: str | None) -> str:
    return (env_name or "").split("/", 1)[-1].split("-", 1)[0].lower()


def _as_array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return np.asarray([], dtype=np.float32)


def _step_observation(step: dict[str, Any]) -> np.ndarray:
    return _as_array(step.get("observation"))


def _step_action(step: dict[str, Any]) -> np.ndarray:
    return _as_array(step.get("action"))


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _sign_change_count(values: list[float], threshold: float = 1e-6) -> int:
    signs: list[int] = []
    for value in values:
        if value > threshold:
            signs.append(1)
        elif value < -threshold:
            signs.append(-1)
        else:
            signs.append(0)
    filtered = [sign for sign in signs if sign != 0]
    if len(filtered) < 2:
        return 0
    return sum(1 for left, right in zip(filtered, filtered[1:]) if left != right)


def _transition_count(values: list[int]) -> int:
    if len(values) < 2:
        return 0
    return sum(1 for left, right in zip(values, values[1:]) if left != right)


def _pair_balance(left_values: list[float], right_values: list[float]) -> float:
    left_mag = _mean([abs(value) for value in left_values])
    right_mag = _mean([abs(value) for value in right_values])
    total = left_mag + right_mag
    if total <= 1e-6:
        return 1.0
    return 1.0 - abs(left_mag - right_mag) / total


def _metric_meta(description: str, tags: list[str], kind: str = "continuous") -> dict[str, Any]:
    return {
        "description": description,
        "tags": tags,
        "kind": kind,
    }


def _add_metric(
    metrics: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    name: str,
    value: float,
    description: str,
    tags: list[str],
    kind: str = "continuous",
) -> None:
    metrics[name] = float(value)
    metadata[name] = _metric_meta(description, tags, kind=kind)


def _infer_series_kind(signal_name: str) -> str:
    lower_name = signal_name.lower()
    if any(token in lower_name for token in ("contact", "touch", "grounded", "leg_contact")):
        return "binary"
    if any(token in lower_name for token in ("quat_",)):
        return "bounded"
    return "continuous"


def _add_series_metrics(
    metrics: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    base_name: str,
    values: list[float],
    *,
    description_root: str,
    tags: list[str],
    kind: str,
) -> None:
    if not values:
        return
    _add_metric(metrics, metadata, f"{base_name}_mean", _mean(values), f"Mean of {description_root}.", [*tags, "mean"], kind=kind)
    _add_metric(metrics, metadata, f"{base_name}_std", _std(values), f"Standard deviation of {description_root}.", [*tags, "variability"], kind=kind)
    _add_metric(metrics, metadata, f"{base_name}_min", min(values), f"Minimum of {description_root}.", [*tags, "minimum"], kind=kind)
    _add_metric(metrics, metadata, f"{base_name}_max", max(values), f"Maximum of {description_root}.", [*tags, "maximum"], kind=kind)
    _add_metric(metrics, metadata, f"{base_name}_range", max(values) - min(values), f"Range of {description_root}.", [*tags, "range"], kind=kind)
    _add_metric(metrics, metadata, f"{base_name}_abs_mean", _mean([abs(value) for value in values]), f"Mean absolute magnitude of {description_root}.", [*tags, "magnitude"], kind=kind)
    _add_metric(metrics, metadata, f"{base_name}_delta", values[-1] - values[0], f"End minus start delta of {description_root}.", [*tags, "delta"], kind=kind)

    if kind == "binary":
        binary_values = [1 if value > 0.5 else 0 for value in values]
        _add_metric(metrics, metadata, f"{base_name}_true_fraction", _mean(binary_values), f"Fraction of steps where {description_root} is true.", [*tags, "fraction", "binary"], kind=kind)
        _add_metric(metrics, metadata, f"{base_name}_transition_count", _transition_count(binary_values), f"Number of true/false transitions for {description_root}.", [*tags, "transitions", "binary"], kind=kind)
        return

    _add_metric(metrics, metadata, f"{base_name}_positive_fraction", _mean([1.0 if value > 0 else 0.0 for value in values]), f"Fraction of steps where {description_root} is positive.", [*tags, "fraction", "positive"], kind=kind)
    _add_metric(metrics, metadata, f"{base_name}_negative_fraction", _mean([1.0 if value < 0 else 0.0 for value in values]), f"Fraction of steps where {description_root} is negative.", [*tags, "fraction", "negative"], kind=kind)
    _add_metric(metrics, metadata, f"{base_name}_near_zero_fraction", _mean([1.0 if abs(value) < 1e-3 else 0.0 for value in values]), f"Fraction of steps where {description_root} stays near zero.", [*tags, "fraction", "near_zero"], kind=kind)
    sign_change_count = _sign_change_count(values)
    _add_metric(metrics, metadata, f"{base_name}_sign_change_count", sign_change_count, f"Number of sign changes for {description_root}.", [*tags, "sign_changes"], kind=kind)
    _add_metric(metrics, metadata, f"{base_name}_sign_change_fraction", sign_change_count / max(1, len(values) - 1), f"Fraction of timesteps with sign changes for {description_root}.", [*tags, "sign_changes", "fraction"], kind=kind)


def _signal_tags(signal_name: str, source: str) -> list[str]:
    lower_name = signal_name.lower()
    tags = [source]
    for token, tag in (
        ("position", "position"),
        ("velocity", "velocity"),
        ("angle", "angle"),
        ("pitch", "pitch"),
        ("yaw", "yaw"),
        ("roll", "roll"),
        ("height", "height"),
        ("contact", "contact"),
        ("left_", "left"),
        ("right_", "right"),
        ("front_", "front"),
        ("back_", "back"),
        ("action", "action"),
        ("target", "target"),
        ("delta", "delta"),
    ):
        if token in lower_name:
            tags.append(tag)
    return sorted(set(tags))


def _named_signal_series(history: list[dict[str, Any]], env_name: str | None) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, str]]:
    """
    Takes in dictionary of history and optional string env_name. Returns tuple containing the following three dictionaries:
    1. observation_series: mapping from observation signal name to list of float values for that signal across the episode.
    2. action_series: mapping from action signal name to list of float values for that signal across the episode.
    3. descriptions: mapping from signal name to human readable description of that signal, if available. Otherwise the description is just the signal name.
    """
    if not history:
        return {}, {}, {}
    first_obs = _step_observation(history[0])
    first_action = _step_action(history[0])
    # Get shape of the observation and action based on the first observation and action from the history
    obs_size = int(first_obs.size) if first_obs.ndim <= 1 else int(first_obs.shape[-1] if first_obs.ndim == 1 else 0)
    action_size = int(first_action.size) if first_action.ndim <= 1 else int(first_action.shape[-1] if first_action.ndim == 1 else 0)
    if first_obs.ndim > 1:
        obs_size = 0
    specs = reward_expression_variable_specs(
        env_name=env_name,
        obs_size=obs_size,
        action_size=max(1, action_size),
        raw_term_keys=[],
    )
    obs_specs = [spec for spec in specs if spec.get("source") == "observation" and str(spec.get("name", "")).startswith("obs_")]
    action_specs = [spec for spec in specs if spec.get("source") == "action" and str(spec.get("name", "")).startswith("action_")]

    observation_series: dict[str, list[float]] = {}
    action_series: dict[str, list[float]] = {}
    descriptions: dict[str, str] = {}

    if first_obs.ndim <= 1:
        for index, spec in enumerate(obs_specs):
            display_name = str(spec.get("display_name") or spec["name"])
            descriptions[display_name] = str(spec.get("description") or display_name)
            observation_series[display_name] = [
                _safe_float(_step_observation(step).reshape(-1)[index]) if _step_observation(step).size > index else 0.0
                for step in history
            ]

    if first_action.ndim <= 1:
        flat_action_len = max(1, int(first_action.size))
        for index in range(flat_action_len):
            spec = action_specs[index] if index < len(action_specs) else {"display_name": f"action_{index}", "description": f"Action component {index}."}
            display_name = str(spec.get("display_name") or f"action_{index}")
            descriptions[display_name] = str(spec.get("description") or display_name)
            action_series[display_name] = [
                _safe_float(_step_action(step).reshape(-1)[index]) if _step_action(step).size > index else 0.0
                for step in history
            ]

    return observation_series, action_series, descriptions


def _image_observation_metrics(history: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    metrics: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    if not history:
        return metrics, metadata
    observation_frames = [_step_observation(step) for step in history if _step_observation(step).size]
    if not observation_frames:
        return metrics, metadata
    flattened = [frame.astype(np.float32).reshape(-1) for frame in observation_frames]
    means = [float(frame.mean()) for frame in flattened]
    stds = [float(frame.std()) for frame in flattened]
    nonzero = [float(np.mean(frame > 0)) for frame in flattened]
    _add_metric(metrics, metadata, "observation_pixel_mean", _mean(means), "Mean pixel intensity across observation frames.", ["observation", "image", "pixels", "mean"])
    _add_metric(metrics, metadata, "observation_pixel_std", _mean(stds), "Mean pixel standard deviation across observation frames.", ["observation", "image", "pixels", "variability"])
    _add_metric(metrics, metadata, "observation_nonzero_fraction", _mean(nonzero), "Fraction of nonzero pixels across observation frames.", ["observation", "image", "pixels", "fraction"])
    first_frame = observation_frames[0]
    if first_frame.ndim >= 3 and first_frame.shape[-1] >= 3:
        channel_means = first_frame.astype(np.float32).mean(axis=tuple(range(first_frame.ndim - 1)))
        channel_history = [frame.astype(np.float32).mean(axis=tuple(range(frame.ndim - 1))) for frame in observation_frames]
        for channel_index, channel_name in enumerate(("red", "green", "blue")):
            values = [float(channel_values[channel_index]) for channel_values in channel_history if len(channel_values) > channel_index]
            _add_metric(metrics, metadata, f"{channel_name}_channel_mean", _mean(values), f"Average {channel_name} channel intensity.", ["observation", "image", channel_name, "mean"])
    return metrics, metadata


def _cartpole_like_metrics(history: list[dict[str, Any]], obs_series: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, dict[str, Any]], list[str]]:
    """
    Cartpole like metrics takes a `history` list of dictionaries of the list of floats.

    It calculates various metrics related to signals in obs_series dictionary. Then it adds the metrics 
    for `left_time_fraction`, `right_time_fraction`, and `sign_change_count` for each signal in the list of signals.

    Returns:
        tuple[dict[str, float], dict[str, dict[str, Any]], list[str]]: A tuple containing the metrics dictionary, metadata dictionary, and labels list.
    """
    metrics: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    labels: list[str] = []
    # Cartpole signals, where the left side is negative and the right side is positive
    # This tests for time on the left, time on the right, and time the sign has changed. 
    for signal_name in ("cart_position", "cart_velocity", "pole_angle", "pole_velocity"):
        values = obs_series.get(signal_name)
        if not values:
            continue
        _add_metric(metrics, metadata, f"{signal_name}_left_time_fraction", _mean([1.0 if value < 0 else 0.0 for value in values]), f"Fraction of steps where {signal_name} is on the negative side.", [signal_name, "left", "fraction"])
        _add_metric(metrics, metadata, f"{signal_name}_right_time_fraction", _mean([1.0 if value > 0 else 0.0 for value in values]), f"Fraction of steps where {signal_name} is on the positive side.", [signal_name, "right", "fraction"])
        _add_metric(metrics, metadata, f"{signal_name}_sign_change_count", _sign_change_count(values), f"Number of sign changes for {signal_name}.", [signal_name, "sign_changes"])
    # For cart position, also check for balance between left and right time, and whether the pole angle changes direction frequently.
    cart_position = obs_series.get("cart_position", [])
    pole_angle = obs_series.get("pole_angle", [])
    if cart_position:
        left_fraction = _mean([1.0 if value < 0 else 0.0 for value in cart_position])
        right_fraction = _mean([1.0 if value > 0 else 0.0 for value in cart_position])
        balance = 1.0 - abs(left_fraction - right_fraction)
        _add_metric(metrics, metadata, "cart_side_balance_score", balance, "How evenly the cart occupies left and right sides.", ["cart_position", "balance", "score"])
        if max(left_fraction, right_fraction) > 0.8:
            labels.append("one_sided_cart_motion")
    if pole_angle and _sign_change_count(pole_angle) < 2:
        labels.append("low_pole_direction_reversal")
    return metrics, metadata, labels


def _mountain_car_metrics(obs_series: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """
    Mountain car metrics takes in dictionary of signal names and values.

    It calculates various metrics related to signals in obs_series dictionary. Then it adds the metrics
    for `hill_progress_fraction` and `velocity_reversal_count` for the position and velocity signals.
    """
    metrics: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    position = obs_series.get("position", [])
    velocity = obs_series.get("velocity", [])
    if position:
        _add_metric(metrics, metadata, "hill_progress_fraction", _mean([1.0 if value > 0 else 0.0 for value in position]), "Fraction of steps spent on the goal-side half of the hill.", ["position", "progress", "fraction"])
    if velocity:
        _add_metric(metrics, metadata, "velocity_reversal_count", _sign_change_count(velocity), "Number of velocity sign reversals.", ["velocity", "reversal"])
    return metrics, metadata


def _pendulum_metrics(obs_series: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    metrics: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    x_values = obs_series.get("x", [])
    y_values = obs_series.get("y", [])
    if x_values and y_values and len(x_values) == len(y_values):
        angles = [float(np.arctan2(y, x)) for x, y in zip(x_values, y_values)]
        _add_series_metrics(metrics, metadata, "pendulum_angle", angles, description_root="pendulum angle reconstructed from x/y", tags=["pendulum", "angle"], kind="continuous")
        _add_metric(metrics, metadata, "upright_half_fraction", _mean([1.0 if y > 0 else 0.0 for y in y_values]), "Fraction of steps spent in the upper half-plane.", ["pendulum", "upright", "fraction"])
    return metrics, metadata


def _acrobot_metrics(obs_series: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    metrics: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for pair_index in (1, 2):
        cos_values = obs_series.get(f"cos_theta{pair_index}", [])
        sin_values = obs_series.get(f"sin_theta{pair_index}", [])
        if cos_values and sin_values and len(cos_values) == len(sin_values):
            angles = [float(np.arctan2(sin_value, cos_value)) for cos_value, sin_value in zip(cos_values, sin_values)]
            _add_series_metrics(metrics, metadata, f"joint{pair_index}_angle", angles, description_root=f"Acrobot joint {pair_index} angle", tags=["acrobot", "angle", f"joint{pair_index}"], kind="continuous")
    return metrics, metadata


def _lunar_lander_metrics(obs_series: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, dict[str, Any]], list[str]]:
    metrics: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    labels: list[str] = []
    for contact_name in ("left_leg_contact", "right_leg_contact"):
        values = obs_series.get(contact_name, [])
        if values:
            _add_series_metrics(metrics, metadata, contact_name, values, description_root=contact_name.replace("_", " "), tags=["lander", "contact", contact_name], kind="binary")
    angle_values = obs_series.get("angle", [])
    if angle_values:
        tilt_fraction = _mean([1.0 if abs(value) > 0.25 else 0.0 for value in angle_values])
        _add_metric(metrics, metadata, "tilted_fraction", tilt_fraction, "Fraction of steps with substantial lander tilt.", ["lander", "tilt", "fraction"])
        if tilt_fraction > 0.6:
            labels.append("persistent_lander_tilt")
    return metrics, metadata, labels


def _reacher_metrics(obs_series: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    metrics: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    dx = obs_series.get("fingertip_delta_x", [])
    dy = obs_series.get("fingertip_delta_y", [])
    dz = obs_series.get("fingertip_delta_z", [])
    if dx and dy and dz and len(dx) == len(dy) == len(dz):
        distances = [float(np.sqrt(x * x + y * y + z * z)) for x, y, z in zip(dx, dy, dz)]
        _add_series_metrics(metrics, metadata, "target_distance", distances, description_root="distance from fingertip to target", tags=["reacher", "target", "distance"], kind="continuous")
    return metrics, metadata


def _paired_left_right_metrics(obs_series: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    metrics: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    keys = list(obs_series.keys())
    for key in keys:
        if "left_" not in key:
            continue
        right_key = key.replace("left_", "right_")
        if right_key not in obs_series:
            continue
        balance = _pair_balance(obs_series[key], obs_series[right_key])
        metric_name = key.replace("left_", "left_right_") + "_balance_score"
        _add_metric(metrics, metadata, metric_name, balance, f"Symmetry score between {key} and {right_key}.", ["symmetry", "left_right", key, right_key, "score"])
    return metrics, metadata


def _body_motion_metrics(obs_series: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    metrics: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for velocity_name in ("x_velocity", "y_velocity", "z_velocity", "pitch_velocity", "roll_velocity", "yaw_velocity", "torso_angular_velocity"):
        values = obs_series.get(velocity_name, [])
        if not values:
            continue
        _add_metric(metrics, metadata, f"{velocity_name}_positive_fraction", _mean([1.0 if value > 0 else 0.0 for value in values]), f"Fraction of steps where {velocity_name} is positive.", [velocity_name, "fraction", "positive"])
        _add_metric(metrics, metadata, f"{velocity_name}_negative_fraction", _mean([1.0 if value < 0 else 0.0 for value in values]), f"Fraction of steps where {velocity_name} is negative.", [velocity_name, "fraction", "negative"])
        _add_metric(metrics, metadata, f"{velocity_name}_sign_change_count", _sign_change_count(values), f"Number of sign changes for {velocity_name}.", [velocity_name, "sign_changes"])
    return metrics, metadata


def _build_episode_metric_library(episode: dict[str, Any], env_name: str | None) -> dict[str, Any]:
    """
    This function builds the metric library for everything within an episode.

        It looks for the behavior trace or reward history to extract time series data. It computes general metrics like episode length and total reward, as well as signal-specific metrics for observations and actions. 
        For certain known environments, it also computes environment-specific metrics. The output includes a dictionary of metric values, metadata describing each metric, and any relevant labels for the episode.

    Args:
        episode: A dictionary containing episode data, including behavior trace or reward history, total reward, and episode length.
        env_name: String name of environment, used to compute environment-specific metrics when possible.

    Returns:
        A dictionary containing metric values, metadata, and labels for the episode.
    """
    history = list(episode.get("behavior_trace") or [])
    if not history:
        history = list(episode.get("reward_history") or [])
    metrics: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    labels: list[str] = []
    episode_length = int(episode.get("episode_terminal_timestep", len(history)))
    # description and tags at then end
    _add_metric(metrics, metadata, "episode_length", episode_length, "Number of timesteps in the episode.", ["episode", "length"])
    _add_metric(metrics, metadata, "episode_reward_total", _safe_float(episode.get("reward", 0.0)), "Total episode reward.", ["episode", "reward", "total"])

    if not history:
        return {"metrics": metrics, "metadata": metadata, "labels": labels}

    first_obs = _step_observation(history[0])
    if first_obs.ndim > 1:
        image_metrics, image_meta = _image_observation_metrics(history)
        metrics.update(image_metrics)
        metadata.update(image_meta)

    obs_series, action_series, descriptions = _named_signal_series(history, env_name)
    for signal_name, values in obs_series.items():
        _add_series_metrics(
            metrics,
            metadata,
            signal_name,
            values,
            description_root=descriptions.get(signal_name, signal_name),
            tags=_signal_tags(signal_name, "observation"),
            kind=_infer_series_kind(signal_name),
        )
    for signal_name, values in action_series.items():
        _add_series_metrics(
            metrics,
            metadata,
            signal_name,
            values,
            description_root=descriptions.get(signal_name, signal_name),
            tags=_signal_tags(signal_name, "action"),
            kind=_infer_series_kind(signal_name),
        )

    prefix = _env_prefix(env_name)
    env_metrics: dict[str, float] = {}
    env_metadata: dict[str, dict[str, Any]] = {}
    env_labels: list[str] = []
    if prefix in {"cartpole", "invertedpendulum", "inverteddoublependulum"}:
        env_metrics, env_metadata, env_labels = _cartpole_like_metrics(history, obs_series)
    elif prefix in {"mountaincar", "mountaincarcontinuous"}:
        env_metrics, env_metadata = _mountain_car_metrics(obs_series)
    elif prefix == "pendulum":
        env_metrics, env_metadata = _pendulum_metrics(obs_series)
    elif prefix == "acrobot":
        env_metrics, env_metadata = _acrobot_metrics(obs_series)
    elif prefix == "lunarlander":
        env_metrics, env_metadata, env_labels = _lunar_lander_metrics(obs_series)
    elif prefix == "reacher":
        env_metrics, env_metadata = _reacher_metrics(obs_series)

    paired_metrics, paired_meta = _paired_left_right_metrics(obs_series)
    body_metrics, body_meta = _body_motion_metrics(obs_series)
    metrics.update(env_metrics)
    metrics.update(paired_metrics)
    metrics.update(body_metrics)
    metadata.update(env_metadata)
    metadata.update(paired_meta)
    metadata.update(body_meta)
    labels.extend(env_labels)

    return {
        "metrics": metrics,
        "metadata": metadata,
        "labels": sorted(set(labels)),
    }


def build_behavior_metric_report(episodes: list[dict[str, Any]], source: str, env_name: str | None = None) -> dict[str, Any]:
    recent = list(episodes or [])[:25]
    # Get per episode reports for the last 25 episodes
    per_episode_reports = [_build_episode_metric_library(episode, env_name) for episode in recent]
    aggregate_metrics: dict[str, list[float]] = defaultdict(list)
    metadata_lookup: dict[str, dict[str, Any]] = {}
    labels = Counter()
    for report in per_episode_reports:
        for metric_name, metric_value in report.get("metrics", {}).items():
            aggregate_metrics[metric_name].append(_safe_float(metric_value))
        metadata_lookup.update(report.get("metadata", {}))
        for label in report.get("labels", []):
            labels[str(label)] += 1

    condensed_metrics = {
        metric_name: _mean(values)
        for metric_name, values in aggregate_metrics.items()
    }
    notable_metrics = sorted(
        condensed_metrics.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:40]
    return {
        "summary": {
            "source": source,
            "env_name": env_name,
            "episodes_analyzed": len(recent),
            "metric_count": len(condensed_metrics),
            "available_labels": dict(labels),
        },
        "metrics": condensed_metrics,
        "metric_metadata": metadata_lookup,
        "notable_metrics": [
            {
                "name": metric_name,
                "value": value,
                "description": metadata_lookup.get(metric_name, {}).get("description", metric_name),
                "tags": metadata_lookup.get(metric_name, {}).get("tags", []),
            }
            for metric_name, value in notable_metrics
        ],
        "episode_metric_samples": [
            {
                "episode": recent[index].get("episode"),
                "metrics": report.get("metrics", {}),
                "labels": report.get("labels", []),
            }
            for index, report in enumerate(per_episode_reports[:5])
        ],
    }
