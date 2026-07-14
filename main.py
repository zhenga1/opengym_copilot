from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import asyncio
import gymnasium as gym
from pydantic import BaseModel
from pyparsing import Optional
import torch
import json
import numpy as np
import cv2
import base64
import os
import re
import socket
import time
from typing import Any
import importlib.util
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request
#import constants

from urllib.parse import parse_qs

import testenv

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

from train_backend_reward_tuning.reward_shaping import (
    RewardShapingWrapper,
    enrich_reward_context_with_task_config,
    evaluate_reward_expression,
    normalize_reward_terms,
    reward_monitor_keys,
    reward_expression_variable_specs,
    reward_template_for_env,
    validate_reward_terms,
)
from deterministic_insights import build_episode_insights
from behavior_metrics import build_behavior_metric_report
from behavior_tags import (
    available_behavior_tags_for_env,
    build_behavior_tag_report,
    heuristic_behavior_tag_plan,
    normalize_behavior_tag_plan,
)
from run_logging import (
    LOGS_DIR,
    configure_backend_logging,
    log_llm_trace,
    log_reward_spec_snapshot,
    log_run_event,
    log_task_variable_names,
)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor
    from training_progress_callback import TrainingProgressCallback
    SB3_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    PPO = None
    DummyVecEnv = None
    BaseCallback = object
    CallbackList = None
    Monitor = None
    TrainingProgressCallback = None
    SB3_IMPORT_ERROR = exc


trained_model_paths = {} # run_id to most recent saved model paths
training_model_devices = {} # run_id to device to use
training_hyperparams_by_run = {} # run_id to ppo/training config
RUNS_TRAINING_STATUS = {}  # run_id -> {"status": "running|done|error", "model_path": str|None, ...}
reward_configs_by_run = {}  # run_id -> {"env_name": str, "terms": list[dict]}
task_configs_by_run = {}  # run_id -> {"env_name": str, "goal": str, "task_params": list[dict], ...}
task_config_request_status_by_run = {}  # run_id -> live LLM request status


app = FastAPI()
env_name = "" # unknown for now
logger = configure_backend_logging()

device = "cuda" if torch.cuda.is_available() else "cpu"
time_intervals = {}
rollout_fps = {}

BASE_DIR = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env", override=False)
DATA_DIR = Path(os.getenv("OPEN_GYM_DATA_DIR", str(BASE_DIR))).resolve()
MODELS_DIR = (DATA_DIR / "models").resolve()
ROLLOUTS_DIR = (DATA_DIR / "rollouts").resolve()
REWARD_CONFIGS_DIR = (DATA_DIR / "reward_configs").resolve()
PROGRESS_LOG_PATH = (DATA_DIR / "progress_bar.log").resolve()
FRONTEND_DIST_DIR = (BASE_DIR / "opengym-frontend" / "dist").resolve()

for path in (DATA_DIR, MODELS_DIR, ROLLOUTS_DIR, REWARD_CONFIGS_DIR):
    path.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

cors_origins_raw = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000",
)
allow_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

progress_bar_log_file = open(PROGRESS_LOG_PATH, "w")
train_run_progresses = {}
HAS_MULTIPART = importlib.util.find_spec("multipart") is not None


class LlmProposalError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage_label: str,
        raw_response_text: str | None = None,
        partial_proposal: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage_label = stage_label
        self.raw_response_text = raw_response_text or ""
        self.partial_proposal = partial_proposal or {}


def ensure_sb3_available():
    if SB3_IMPORT_ERROR is not None:
        raise RuntimeError(
            "stable_baselines3 is required for training and model loading, "
            f"but is not installed in this Python environment: {SB3_IMPORT_ERROR}"
        )

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, runId, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.runId = runId
    
    def _on_step(self) -> bool:
        global train_run_progress
        # For vectorized envs, num_timesteps reflects true progress; n_calls undercounts by n_envs.
        steps_done = int(getattr(self.model, "num_timesteps", 0)) # type: ignore[attr-defined]
        pct = 100 * min(steps_done, self.total_timesteps) / self.total_timesteps
        # file.write(f"Progress: {pct:.2f}%\n")
        # file.write(f"n_calls: {self.n_calls}\n total_timesteps: {self.total_timesteps}\n")
        # file.flush()
        print(f"Progress: {pct:.2f}%", end='\r') # or send to the frontend
        
        train_run_progresses[self.runId] = pct
        return True

# Example helpers (safe from a background thread):
def make_tick_sender(ws_manager):
    """Return a function that schedules a tick to be broadcast to all clients."""
    def _send_tick(payload: dict, run_id: str, topic:str="training"):
        # e.g., ws_manager.broadcast_json(payload)  (thread-safe or queue-based)
        ws_manager.enqueue_json(topic, payload, run_id=run_id)
    return _send_tick

def make_frame_sender(ws_manager, encoder):
    """Return a function that schedules an encoded frame send."""
    def _send_frame(frame_np, run_id: str, topic:str="training"):
        # encode to JPEG/PNG bytes, then enqueue
        ws_manager.enqueue_bytes(topic, encoder(frame_np), run_id=run_id)
    return _send_frame
    
from fastapi import WebSocket, APIRouter, WebSocketDisconnect
from threading import Thread, Lock
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def build_default_model_path(selected_env_name: str) -> str:
    # This builds a string path to a file 
    safe_env_name = (selected_env_name or "env").replace("/", "_")
    return str((MODELS_DIR / f"ppo_model_{safe_env_name}_{timestamp}.zip").resolve())


def build_default_model_filename(selected_env_name: str, run_id: str | None = None) -> str:
    # This builds a string path to a file and replaces any invalid characters with _
    safe_env_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", (selected_env_name or "env").strip() or "env")
    return sanitize_storage_name(f"ppo_model_{safe_env_name}_{run_id or timestamp}", ".zip")


def sanitize_storage_name(name: str | None, suffix: str) -> str:
    # removes leading / training whitespace from "name" parameter
    # and assigns it to raw_name variable
    # then gives a default name if the candidate name (Path(raw_name).name) is empty after stripping
    raw_name = (name or "").strip()
    # gets the CLEAN name from the raw name without the whitespaces
    candidate = Path(raw_name).name
    if not candidate:
        candidate = f"default{suffix}"
    if not candidate.endswith(suffix):
        candidate = f"{candidate}{suffix}"
    return candidate


def resolve_model_output_path(requested_path: str | None, run_id: str | None, selected_env_name: str) -> Path:
    # Called to get a path to save the model to
    default_name = build_default_model_filename(selected_env_name, run_id or "session")
    if not requested_path or not requested_path.strip():
        return (MODELS_DIR / default_name).resolve()

    candidate = Path(requested_path.strip())
    # tries to resolve the absolute path 
    if candidate.is_absolute():
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = (MODELS_DIR / default_name).resolve()
        else:
            # run this if resolved succeeded. 
            try:
                # check to see if the raw resolved path is WITHIN the MODELS_DIR and throw the ValueError if not. 
                resolved.relative_to(MODELS_DIR)
            except ValueError:
                resolved = (MODELS_DIR / sanitize_storage_name(candidate.name, ".zip")).resolve()
    else:
        # basically resolve it relative to MODELS_DIR
        safe_parts = [part for part in candidate.parts if part not in {"", ".", ".."}]
        if safe_parts and safe_parts[0].lower() == "models":
            # forces a fix if the first part is "models", to avoid "models/models.zip"
            safe_parts = safe_parts[1:]
        resolved = (MODELS_DIR / Path(*safe_parts)).resolve() if safe_parts else (MODELS_DIR / default_name).resolve()

    # then ensures the path ends with a ".zip" suffix
    if resolved.suffix.lower() != ".zip":
        resolved = resolved.with_suffix(".zip")
    try:
        resolved.relative_to(MODELS_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="training output path must stay inside the managed models directory") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _parse_model_timestamp_from_name(filename: str) -> datetime | None:
    matches = re.findall(r"(\d{8}_\d{6})", filename)
    for value in reversed(matches):
        try:
            return datetime.strptime(value, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
    return None


def _infer_env_name_from_model_name(filename: str) -> str | None:
    stem = Path(filename).stem
    if stem.startswith("ppo_model_"):
        remainder = stem.removeprefix("ppo_model_")
        timestamp_match = re.search(r"_\d{8}_\d{6}$", remainder)
        if timestamp_match:
            return remainder[:timestamp_match.start()] or None
        run_match = re.search(r"_[0-9a-fA-F-]{6,}$", remainder)
        if run_match:
            return remainder[:run_match.start()] or None
        return remainder or None
    return None


def list_model_records() -> list[dict[str, Any]]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for path in MODELS_DIR.iterdir():
        if not path.is_file() or path.suffix != ".zip":
            continue
        stat = path.stat()
        parsed_dt = _parse_model_timestamp_from_name(path.name)
        created_ts = parsed_dt.timestamp() if parsed_dt is not None else float(stat.st_mtime)
        created_at = (
            parsed_dt.isoformat(timespec="seconds")
            if parsed_dt is not None
            else datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        )
        metadata = read_model_metadata(path)
        env_guess = str(metadata.get("env_name") or _infer_env_name_from_model_name(path.name) or "").strip() or None
        records.append({
            "name": path.name,
            "path": str(path.resolve()),
            "created_at": created_at,
            "created_ts": created_ts,
            "size_bytes": int(stat.st_size),
            "env_name": env_guess,
            "is_temp": path.name.startswith("temp_"),
            "display_name": env_guess or path.stem,
            "metadata_source": metadata.get("source"),
        })
    return records


def model_metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(".meta.json")


def write_model_metadata(
    model_path: Path,
    *,
    env_name: str,
    run_id: str | None = None,
    source: str = "training",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "model_name": model_path.name,
        "env_name": str(env_name).strip(),
        "run_id": run_id,
        "source": source,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "details": details or {},
    }
    with open(model_metadata_path(model_path), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return metadata


def read_model_metadata(model_path: Path) -> dict[str, Any]:
    metadata_file = model_metadata_path(model_path)
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
        except (OSError, json.JSONDecodeError):
            pass
    inferred_env = _infer_env_name_from_model_name(model_path.name)
    return {
        "model_name": model_path.name,
        "env_name": inferred_env,
        "run_id": None,
        "source": "inferred",
        "saved_at": None,
        "details": {},
    }

def encode_jpeg(frame_np, quality=80):
    # encode the frame_np to JPEG bytes with the given quality setting
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame_np).save(buf, format="jpeg", quality=quality)
    return buf.getvalue()


class RewardTermPayload(BaseModel):
    # Payload for a single reward term in the reward configuration
    key: str
    weight: float
    enabled: bool = True
    label: str | None = None
    description: str | None = None
    expression: str | None = None


class RewardConfigUpdateRequest(BaseModel):
    #  Payload for updating the reward configuration
    run_id: str
    env_name: str
    terms: list[RewardTermPayload]
    goal: str | None = None
    task_params: list[dict[str, Any]] = []
    derived_signals: list[dict[str, Any]] = []
    success_metric: str | None = None
    rationale: str | None = None
    warnings: list[str] = []
    provider: str | None = None
    model: str | None = None
    behavior_plan: dict[str, Any] | None = None


class TaskConfigProposalRequest(BaseModel):
    # Payload for a task configuration proposal
    run_id: str
    env_name: str
    goal: str
    llm_id: str | None = None


class TaskConfigApplyRequest(BaseModel):
    # Payload for applying a task configuration proposal
    run_id: str
    env_name: str
    goal: str | None = None
    task_params: list[dict[str, Any]] = []
    derived_signals: list[dict[str, Any]] = []
    reward_terms: list[RewardTermPayload]
    success_metric: str | None = None
    rationale: str | None = None
    warnings: list[str] = []
    provider: str | None = None
    model: str | None = None
    behavior_plan: dict[str, Any] | None = None


class SaveRewardConfigRequest(BaseModel):
    # Payload for saving a reward configuration
    run_id: str
    env_name: str
    filename: str | None = None
    source_type: str | None = None


class LoadRewardConfigRequest(BaseModel):
    # Payload for loading a reward configuration
    run_id: str
    env_name: str
    filename: str


def update_task_config_request_status(run_id: str | None, **fields: Any) -> None:
    # updates the task config request status
    # for the given run_id 
    if not run_id:
        return
    # see if the run_id is already in the task_config_request_status_by_run
    current = task_config_request_status_by_run.get(run_id, {})
    # if not save the current as default status. 
    if not current:
        current = {
            "run_id": run_id,
            "status": "idle",
            "message": "No active proposal request.",
            "attempt": 0,
            "elapsed_sec": 0.0,
        }
    current.update(fields)
    task_config_request_status_by_run[run_id] = current


DEFAULT_TRAINING_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "lr_schedule": "constant",
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "model_size": "medium",
}

MODEL_SIZE_TO_NET_ARCH = {
    "small": [64, 64],
    "medium": [128, 128],
    "large": [256, 256],
}
reward_variable_cache: dict[str, list[dict[str, Any]]] = {}


def reward_variable_specs_for_env(env_name: str) -> list[dict[str, Any]]:
    # Take env name and return the cached reward specifications
    # if exist, return from the cache. 
    """
    The function calculates the size of the observation space and action space. If the action space has a shape attribute, it calculates the size by multiplying the dimensions of the shape. Otherwise, it sets the size to 1.

    Next, it retrieves the keys of the reward template for the given env_name and passes them along with the calculated sizes to the 
    reward_expression_variable_specs function.

    The resulting specs are stored in the reward_variable_cache with the env_name as the key. 
    Finally, the function returns the specs and ensures that the environment is closed in a finally block.
    """
    cached = reward_variable_cache.get(env_name)
    if cached is not None:
        return cached

    # otherwise, get specs directly from the environment
    env = gym.make(env_name)
    try:
        # get the observation and action spaces
        obs_space = env.observation_space
        action_space = env.action_space

        # get the observation and action sizes
        obs_size = int(np.prod(obs_space.shape)) if getattr(obs_space, "shape", None) else 1
        if getattr(action_space, "shape", None):
            action_size = int(np.prod(action_space.shape))
        else:
            action_size = 1

        # calculate the reward variable specs. Get deepcopy of the proper template. 
        # "key" is the name of the reward term 
        raw_term_keys = [term["key"] for term in reward_template_for_env(env_name)]

        # calculate the reward variable specs
        specs = reward_expression_variable_specs(
            env_name=env_name,
            obs_size=obs_size,
            action_size=action_size,
            raw_term_keys=raw_term_keys,
        )
        # store the specs in the cache before returning
        reward_variable_cache[env_name] = specs
        return specs
    finally:
        # close the environment to free up resources
        env.close()


def reward_variable_names_for_env(env_name: str) -> list[str]:
    
    variable_names: list[str] = []
    seen: set[str] = set()
    for spec in reward_variable_specs_for_env(env_name):
        for variable_name in [spec["name"], *spec.get("aliases", [])]:
            if variable_name and variable_name not in seen:
                seen.add(variable_name)
                variable_names.append(variable_name)
    return variable_names


def reward_formula_examples_for_env(env_name: str) -> list[str]:
    prefix = env_name.split("/", 1)[-1].split("-", 1)[0].lower()
    if prefix == "cartpole":
        return [
            "0.2 * sin(theta)",
            "-0.05 * cart_velocity * cart_velocity",
            "clip(native + 0.1 * survival_bonus - 0.2 * abs(pole_angle), -10, 10)",
        ]
    if prefix == "pendulum":
        return [
            "0.25 * y",
            "-0.05 * angular_velocity * angular_velocity",
            "clip(native + 0.2 * y - 0.05 * abs(theta_dot), -10, 10)",
        ]
    return [
        "0.5 * observation_0 * observation_0",
        "-abs(native_reward)",
        "clip(native_reward, -10, 10)",
    ]


def get_task_config_for_run(run_id: str | None, env_name: str) -> dict[str, Any]:
    if not run_id:
        return {
            "env_name": env_name,
            "goal": "",
            "task_params": [],
            "derived_signals": [],
            "success_metric": "",
            "rationale": "",
            "warnings": [],
            "provider": "none",
            "model": "",
            "behavior_plan": {},
        }
    current = task_configs_by_run.get(run_id)
    if current is None or current.get("env_name") != env_name:
        task_configs_by_run[run_id] = {
            "env_name": env_name,
            "goal": "",
            "task_params": [],
            "derived_signals": [],
            "success_metric": "",
            "rationale": "",
            "warnings": [],
            "provider": "none",
            "model": "",
            "behavior_plan": {},
        }
    return task_configs_by_run[run_id]


def set_task_config_for_run(run_id: str, env_name: str, config: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "env_name": env_name,
        "goal": str(config.get("goal") or "").strip(),
        "task_params": list(config.get("task_params") or []),
        "derived_signals": list(config.get("derived_signals") or []),
        "success_metric": str(config.get("success_metric") or "").strip(),
        "rationale": str(config.get("rationale") or "").strip(),
        "warnings": [str(item) for item in (config.get("warnings") or []) if str(item).strip()],
        "provider": str(config.get("provider") or "manual").strip(),
        "model": str(config.get("model") or "").strip(),
        "behavior_plan": normalize_behavior_tag_plan(
            config.get("behavior_plan"),
            env_name,
            goal=str(config.get("goal") or "").strip(),
        ),
    }
    task_configs_by_run[run_id] = normalized
    return normalized


def validate_task_config_for_run(config: dict[str, Any], env_name: str, run_id: str | None = None) -> dict[str, Any]:
    base_names = reward_variable_names_for_env(env_name)
    param_names: list[str] = []
    param_values: dict[str, float] = {}
    seen_param_names: set[str] = set()
    for param in config.get("task_params", []) or []:
        key = str(param.get("key") or "").strip()
        if not key:
            continue
        if key in seen_param_names:
            raise ValueError(f"Duplicate task parameter '{key}'.")
        seen_param_names.add(key)
        param_names.append(key)
        try:
            param_values[key] = float(param.get("value", 0.0))
        except (TypeError, ValueError):
            param_values[key] = 0.0

    derived_names: list[str] = []
    seen_signal_names: set[str] = set()
    zero_context = {name: 0.0 for name in base_names}
    # update the context with the updated task parameters
    zero_context.update(param_values)
    for signal in config.get("derived_signals", []) or []:
        key = str(signal.get("key") or "").strip()
        expression = str(signal.get("expression") or "").strip()
        if not key:
            continue
        if key in seen_signal_names:
            raise ValueError(f"Duplicate derived signal '{key}'.")
        seen_signal_names.add(key)
        if expression:
            evaluate_reward_expression(expression, zero_context)
        zero_context[key] = 0.0
        derived_names.append(key)

    validate_reward_terms(
        env_name,
        config.get("reward_terms") or [],
        available_variable_names=[*base_names, *param_names, *derived_names],
    )
    return config


def normalize_task_config_proposal_shape(proposal: dict[str, Any] | None) -> dict[str, Any]:
    current = dict(proposal or {})
    dropped_messages: list[str] = []

    def _normalize_object_list(value: Any, field_name: str) -> list[dict[str, Any]]:
        source_items: list[Any]
        if isinstance(value, list):
            source_items = value
        elif isinstance(value, dict):
            if field_name == "task_params":
                source_items = [
                    {"key": key, "value": item}
                    for key, item in value.items()
                ]
            elif field_name == "derived_signals":
                source_items = [
                    {
                        "key": key,
                        "expression": item if isinstance(item, str) else (item.get("expression") if isinstance(item, dict) else ""),
                        "description": item.get("description", "") if isinstance(item, dict) else "",
                    }
                    for key, item in value.items()
                ]
            else:
                dropped_messages.append(f"Ignored object-form field '{field_name}' from model output.")
                return []
        else:
            if value not in (None, ""):
                dropped_messages.append(f"Ignored non-list field '{field_name}' from model output.")
            return []
        normalized_items: list[dict[str, Any]] = []
        for index, item in enumerate(source_items):
            if isinstance(item, dict):
                normalized_item = dict(item)
                if field_name == "task_params":
                    normalized_item["key"] = normalized_item.get("key", normalized_item.get("name", ""))
                    normalized_item["description"] = normalized_item.get("description", "")
                elif field_name == "derived_signals":
                    normalized_item["key"] = normalized_item.get("key", normalized_item.get("name", ""))
                    normalized_item["expression"] = normalized_item.get("expression", normalized_item.get("formula", ""))
                    normalized_item["description"] = normalized_item.get("description", "")
                elif field_name == "reward_terms":
                    normalized_item["key"] = normalized_item.get("key", normalized_item.get("name", ""))
                    normalized_item["label"] = normalized_item.get("label", normalized_item.get("title", normalized_item.get("key", "")))
                    normalized_item["description"] = normalized_item.get("description", "")
                    normalized_item["expression"] = normalized_item.get("expression", normalized_item.get("formula", ""))
                normalized_items.append(normalized_item)
                continue
            if isinstance(item, str):
                text = item.strip()
                if text:
                    dropped_messages.append(
                        f"Ignored string item in '{field_name}' at index {index}; expected an object with named fields."
                    )
                continue
            if item is not None:
                dropped_messages.append(
                    f"Ignored unsupported item in '{field_name}' at index {index}; expected an object."
                )
        return normalized_items

    current["task_params"] = _normalize_object_list(current.get("task_params"), "task_params")
    current["derived_signals"] = _normalize_object_list(current.get("derived_signals"), "derived_signals")
    current["reward_terms"] = _normalize_object_list(current.get("reward_terms"), "reward_terms")

    warnings_value = current.get("warnings")
    if isinstance(warnings_value, list):
        current["warnings"] = [str(item).strip() for item in warnings_value if str(item).strip()]
    elif isinstance(warnings_value, str):
        current["warnings"] = [warnings_value.strip()] if warnings_value.strip() else []
    else:
        current["warnings"] = []

    if dropped_messages:
        current["warnings"] = [*dropped_messages, *current["warnings"]]
    return current


def task_variable_specs_for_run(run_id: str | None, env_name: str) -> list[dict[str, Any]]:
    specs = [dict(spec) for spec in reward_variable_specs_for_env(env_name)]
    task_config = get_task_config_for_run(run_id, env_name)
    for param in task_config.get("task_params", []):
        key = str(param.get("key") or "").strip()
        if not key:
            continue
        specs.append({
            "name": key,
            "source": "task_param",
            "display_name": key,
            "description": str(param.get("description") or f"Task parameter '{key}'."),
            "aliases": [],
        })
    for signal in task_config.get("derived_signals", []):
        key = str(signal.get("key") or "").strip()
        if not key:
            continue
        specs.append({
            "name": key,
            "source": "derived_signal",
            "display_name": key,
            "description": str(signal.get("description") or f"Derived signal '{key}'."),
            "aliases": [],
        })
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        name = str(spec.get("name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            deduped.append(spec)
    return deduped


def task_variable_names_for_run(run_id: str | None, env_name: str) -> list[str]:
    variable_names: list[str] = []
    seen: set[str] = set()
    for spec in task_variable_specs_for_run(run_id, env_name):
        for variable_name in [spec["name"], *spec.get("aliases", [])]:
            if variable_name and variable_name not in seen:
                seen.add(variable_name)
                variable_names.append(variable_name)
    log_task_variable_names(
        run_id=run_id,
        env_name=env_name,
        variable_names=variable_names,
        source="task_variable_names_for_run",
        details={
            "task_param_count": len(get_task_config_for_run(run_id, env_name).get("task_params") or []),
            "derived_signal_count": len(get_task_config_for_run(run_id, env_name).get("derived_signals") or []),
        },
    )
    return variable_names


def _compact_behavior_tag_catalog(env_name: str) -> list[dict[str, Any]]:
    compact_catalog: list[dict[str, Any]] = []
    for tag in available_behavior_tags_for_env(env_name):
        compact_catalog.append({
            "key": str(tag.get("key") or "").strip(),
            "polarity": str(tag.get("polarity") or "").strip(),
            "title": str(tag.get("title") or "").strip(),
            "tags": [str(item).strip() for item in (tag.get("tags") or []) if str(item).strip()][:4],
        })
    return compact_catalog


def _compact_variable_specs_for_prompt(env_name: str) -> list[dict[str, Any]]:
    compact_specs: list[dict[str, Any]] = []
    for spec in task_variable_specs_for_run(None, env_name):
        aliases = [str(alias).strip() for alias in (spec.get("aliases") or []) if str(alias).strip()]
        compact_specs.append({
            "name": str(spec.get("name") or "").strip(),
            "source": str(spec.get("source") or "").strip(),
            "aliases": aliases[:4],
        })
    return compact_specs


def _compact_reward_terms_for_prompt(env_name: str) -> list[dict[str, Any]]:
    compact_terms: list[dict[str, Any]] = []
    for term in reward_template_for_env(env_name):
        compact_terms.append({
            "key": str(term.get("key") or "").strip(),
            "weight": float(term.get("weight", 0.0)),
            "enabled": bool(term.get("enabled", True)),
            "expression": str(term.get("expression") or "").strip(),
        })
    return compact_terms


def _extract_first_number(text: str, default: float) -> float:
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return float(default)
    try:
        return float(match.group(1))
    except ValueError:
        return float(default)

"""
Takes string and returns JSON object extracted


"""
def _extract_balanced_json_object(text: str) -> str | None:
    source = str(text or "").strip()
    if not source:
        return None
    if source.startswith("```"):
        source = re.sub(r"^```(?:json)?\s*", "", source)
        source = re.sub(r"\s*```$", "", source)
    start = source.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(source)):
        char = source[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    return None


def build_heuristic_task_config_proposal(goal: str, env_name: str) -> dict[str, Any]:
    normalized_goal = str(goal or "").strip()
    lower_goal = normalized_goal.lower()
    terms = reward_template_for_env(env_name)
    behavior_plan = normalize_behavior_tag_plan(
        heuristic_behavior_tag_plan(goal, env_name),
        env_name,
        goal=goal,
    )
    task_params: list[dict[str, Any]] = []
    derived_signals: list[dict[str, Any]] = []
    warnings: list[str] = []
    rationale = "Heuristic fallback proposal based on the current environment schema and goal keywords."
    success_metric = "Long episodes with stable shaped reward and fewer late-episode failure spikes."

    """
    Currently the heuristic proposal only applies to CartPole environments.

    TODO: expand to other environments and goals as needed, and make this more robust to different keyword variations.
    """
    if env_name.startswith("CartPole-") and any(keyword in lower_goal for keyword in ["sway", "swing", "oscillat", "left and right", "frequency", "periodic"]):
        frequency = 0.8
        amplitude = 0.18
        hz_match = re.search(r"(\d+(?:\.\d+)?)\s*hz", lower_goal)
        amp_match = re.search(r"(?:amp(?:litude)?|angle)\s*(\d+(?:\.\d+)?)", lower_goal)
        if hz_match:
            frequency = float(hz_match.group(1))
        if amp_match:
            amplitude = float(amp_match.group(1))
        task_params = [
            {"key": "target_frequency_hz", "value": frequency, "description": "Target sway frequency in Hertz."},
            {"key": "target_amplitude", "value": amplitude, "description": "Target pole sway amplitude in radians."},
        ]
        derived_signals = [
            {
                "key": "reference_angle",
                "expression": f"{amplitude:.6f} * sin(2 * 3.14159265 * {frequency:.6f} * time_sec)",
                "description": "Target periodic pole angle reference.",
            },
            {
                "key": "tracking_error",
                "expression": "pole_angle - reference_angle",
                "description": "Difference between pole angle and target reference.",
            },
        ]
        terms = [
            {
                "key": "native",
                "label": "Native Reward",
                "description": "Keep some native survivability signal.",
                "weight": 0.25,
                "enabled": True,
                "expression": "native_reward",
            },
            {
                "key": "survival_bonus",
                "label": "Survival Bonus",
                "description": "Keep the episode alive while oscillating.",
                "weight": 0.4,
                "enabled": True,
                "expression": "1.0",
            },
            {
                "key": "reference_tracking_penalty",
                "label": "Reference Tracking Penalty",
                "description": "Penalize deviation from the target sway trajectory.",
                "weight": 2.0,
                "enabled": True,
                "expression": f"-abs(pole_angle - ({amplitude:.6f} * sin(2 * 3.14159265 * {frequency:.6f} * time_sec)))",
            },
            {
                "key": "cart_position_penalty",
                "label": "Cart Position Penalty",
                "description": "Keep the cart from running away while swaying.",
                "weight": 0.35,
                "enabled": True,
                "expression": "-square(cart_position)",
            },
            {
                "key": "action_change_penalty",
                "label": "Action Change Penalty",
                "description": "Encourage smoother oscillatory control.",
                "weight": 0.04,
                "enabled": True,
                "expression": "-abs(action_0 - prev_action_0)",
            },
        ]
        success_metric = "Long episodes with bounded cart_position and low mean absolute tracking error to the sway reference."
        rationale = "The goal mentions periodic sway, so the proposal adds a reference-tracking term over time_sec and keeps stabilization penalties to prevent cart runaway."
    else:
        if "smooth" in lower_goal:
            for term in terms:
                if term["key"] == "action_change_penalty":
                    term["weight"] = max(float(term.get("weight", 0.0)), 0.08)
        if any(keyword in lower_goal for keyword in ["upright", "balance", "stable"]):
            for term in terms:
                if term["key"] in {"pole_angle_penalty", "upright_bonus", "stability_penalty"}:
                    term["weight"] = max(float(term.get("weight", 0.0)), 1.2 if "bonus" not in term["key"] else 1.0)
        if any(keyword in lower_goal for keyword in ["fast", "forward", "run", "walk", "hop"]):
            for term in terms:
                if term["key"] in {"forward_bonus", "speed_bonus", "hill_progress_bonus"}:
                    term["weight"] = max(float(term.get("weight", 0.0)), 1.0)
        warnings.append("This fallback proposal only adjusts currently supported reward expressions. Task parameters and derived signals are advisory unless the terms reference supported runtime variables like time_sec or step.")

    return {
        "goal": normalized_goal,
        "behavior_plan": behavior_plan,
        "task_params": task_params,
        "derived_signals": derived_signals,
        "reward_terms": terms,
        "success_metric": success_metric,
        "rationale": rationale,
        "warnings": warnings,
        "provider": "heuristic",
        "model": "",
    }


def _task_config_llm_catalog() -> list[dict[str, Any]]:
    custom_key = os.getenv("TASK_CONFIG_LLM_API_KEY", "").strip()
    glm_key = os.getenv("ZAI_API_KEY", "").strip() or os.getenv("BIGMODEL_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    return [
        {
            "id": "glm",
            "label": "GLM",
            "provider": "glm",
            "description": "Z.ai / BigModel OpenAI-compatible endpoint.",
            "available": bool(glm_key),
            "missing_reason": "" if glm_key else "ZAI_API_KEY or BIGMODEL_API_KEY not provided.",
            "api_key": glm_key,
            "base_url": os.getenv("GLM_BASE_URL", "https://api.z.ai/api/paas/v4").strip().rstrip("/"),
            "model": os.getenv("GLM_MODEL", "glm-5").strip() or "glm-5",
        },
        {
            "id": "openai",
            "label": "OpenAI",
            "provider": "openai-compatible",
            "description": "OpenAI chat-completions compatible task-config planner.",
            "available": bool(openai_key),
            "missing_reason": "" if openai_key else "OPENAI_API_KEY not provided.",
            "api_key": openai_key,
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/"),
            "model": os.getenv("OPENAI_TASK_CONFIG_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini",
        },
        {
            "id": "custom",
            "label": "Custom",
            "provider": "openai-compatible",
            "description": "User-configured OpenAI-compatible endpoint from TASK_CONFIG_LLM_* env vars.",
            "available": bool(custom_key),
            "missing_reason": "" if custom_key else "TASK_CONFIG_LLM_API_KEY not provided.",
            "api_key": custom_key,
            "base_url": os.getenv("TASK_CONFIG_LLM_BASE_URL", "https://api.z.ai/api/paas/v4").strip().rstrip("/"),
            "model": os.getenv("TASK_CONFIG_LLM_MODEL", "glm-5").strip() or "glm-5",
        },
    ]


def _task_config_llm_settings(llm_id: str | None = None) -> dict[str, Any]:
    catalog = _task_config_llm_catalog()
    preferred_id = str(llm_id or os.getenv("TASK_CONFIG_LLM_DEFAULT", "")).strip().lower()
    selected = next((item for item in catalog if item["id"] == preferred_id), None) if preferred_id else None
    if selected is None:
        selected = next((item for item in catalog if item["available"]), None)
    if selected is None:
        raise RuntimeError("No task-config LLM API key is configured. Set ZAI_API_KEY, BIGMODEL_API_KEY, OPENAI_API_KEY, or TASK_CONFIG_LLM_API_KEY.")

    api_key = str(selected.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError(str(selected.get("missing_reason") or f"No API key available for llm_id='{selected['id']}'."))

    timeout_sec = max(15, int(float(os.getenv("TASK_CONFIG_LLM_TIMEOUT_SEC", "120"))))
    max_retries = max(1, int(os.getenv("TASK_CONFIG_LLM_MAX_RETRIES", "2")))
    return {
        "id": selected["id"],
        "label": selected["label"],
        "provider": selected["provider"],
        "api_key": api_key,
        "base_url": str(selected["base_url"]),
        "model": str(selected["model"]),
        "timeout_sec": timeout_sec,
        "max_retries": max_retries,
        "behavior_plan_max_tokens": max(128, int(os.getenv("TASK_CONFIG_BEHAVIOR_PLAN_MAX_TOKENS", "320"))),
        "reward_config_max_tokens": max(384, int(os.getenv("TASK_CONFIG_REWARD_CONFIG_MAX_TOKENS", "1200"))),
    }


def _call_llm_json_response(
    *,
    llm_id: str | None,
    provider: str | None,
    base_url: str,
    model: str,
    api_key: str,
    timeout_sec: int,
    max_retries: int,
    system_prompt: str,
    user_payload: dict[str, Any],
    run_id: str | None,
    env_name: str | None,
    goal: str | None,
    stage_label: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    user_payload_text = json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
    prompt = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload_text},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_completion_tokens": max_output_tokens,
    }
    log_llm_trace(
        run_id=run_id,
        env_name=env_name,
        stage=stage_label,
        trace_type="request_prepared",
        llm_id=llm_id,
        provider=provider,
        model=model,
        base_url=base_url,
        goal=goal,
        request_payload=prompt,
        details={
            "timeout_sec": timeout_sec,
            "max_retries": max_retries,
            "max_output_tokens": max_output_tokens,
            "request_bytes": len(user_payload_text.encode("utf-8")),
        },
    )
    request = urllib_request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(prompt).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    started_at = time.time()
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        update_task_config_request_status(
            run_id,
            status=f"{stage_label}_sending",
            message=f"Sending {stage_label} request to {base_url} with model={model} (attempt {attempt}/{max_retries}).",
            attempt=attempt,
            elapsed_sec=round(time.time() - started_at, 2),
        )
        log_llm_trace(
            run_id=run_id,
            env_name=env_name,
            stage=stage_label,
            trace_type="request_sent",
            llm_id=llm_id,
            provider=provider,
            model=model,
            base_url=base_url,
            goal=goal,
            attempt=attempt,
            request_payload=prompt,
        )
        try:
            with urllib_request.urlopen(request, timeout=timeout_sec) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            log_llm_trace(
                run_id=run_id,
                env_name=env_name,
                stage=stage_label,
                trace_type="http_error",
                llm_id=llm_id,
                provider=provider,
                model=model,
                base_url=base_url,
                goal=goal,
                attempt=attempt,
                request_payload=prompt,
                error=detail,
            )
            update_task_config_request_status(
                run_id,
                status=f"{stage_label}_http_error",
                message=f"{stage_label} HTTP error for model={model}: {detail[:300]}",
                attempt=attempt,
                elapsed_sec=round(time.time() - started_at, 2),
            )
            raise RuntimeError(
                f"{stage_label} request failed for model={model} base_url={base_url}: {detail}"
            ) from exc
        except (urllib_error.URLError, TimeoutError, socket.timeout) as exc:
            last_error = exc
            if attempt >= max_retries:
                reason = getattr(exc, "reason", None) or str(exc)
                log_llm_trace(
                    run_id=run_id,
                    env_name=env_name,
                    stage=stage_label,
                    trace_type="timeout_error",
                    llm_id=llm_id,
                    provider=provider,
                    model=model,
                    base_url=base_url,
                    goal=goal,
                    attempt=attempt,
                    request_payload=prompt,
                    error=str(reason),
                )
                update_task_config_request_status(
                    run_id,
                    status=f"{stage_label}_timeout",
                    message=f"{stage_label} failed after {attempt} attempt(s) for model={model} at {base_url}.",
                    attempt=attempt,
                    elapsed_sec=round(time.time() - started_at, 2),
                )
                raise RuntimeError(
                    f"{stage_label} request timed out/failed after {attempt} attempt(s) "
                    f"for model={model} base_url={base_url} timeout={timeout_sec}s: {reason}"
                ) from exc
            time.sleep(min(2 * attempt, 5))
    else:
        raise RuntimeError(f"{stage_label} request failed for model={model} base_url={base_url}: {last_error}")

    output_text = None
    choices = body.get("choices") or []
    if choices:
        output_text = choices[0].get("message", {}).get("content")
    if not output_text:
        log_llm_trace(
            run_id=run_id,
            env_name=env_name,
            stage=stage_label,
            trace_type="empty_response_error",
            llm_id=llm_id,
            provider=provider,
            model=model,
            base_url=base_url,
            goal=goal,
            request_payload=prompt,
            parsed_response=body if isinstance(body, dict) else {},
            error="No structured output text in first choice.",
        )
        raise RuntimeError(f"{stage_label} response did not include structured output text.")
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        """
        Attempt to recover valid JSON from the output text in case the model included 
        extra commentary or formatting around the JSON. This can help mitigate some common 
        failure modes where the model tries to be helpful by adding explanations or formatting 
        that unfortunately breaks strict JSON parsing.
        """
        recovered_candidate = _extract_balanced_json_object(output_text)
        if recovered_candidate and recovered_candidate != output_text:
            try:
                parsed = json.loads(recovered_candidate)
            except json.JSONDecodeError:
                parsed = None
            else:
                if isinstance(parsed, dict):
                    parsed["_raw_model_response"] = output_text
                    parsed["_parse_recovered"] = True
                    log_llm_trace(
                        run_id=run_id,
                        env_name=env_name,
                        stage=stage_label,
                        trace_type="parse_recovered",
                        llm_id=llm_id,
                        provider=provider,
                        model=model,
                        base_url=base_url,
                        goal=goal,
                        request_payload=prompt,
                        raw_response_text=output_text,
                        parsed_response=parsed,
                        details={"recovered_candidate": recovered_candidate},
                    )
                    update_task_config_request_status(
                        run_id,
                        status=f"{stage_label}_completed",
                        message=f"{stage_label} completed with recovered JSON using model={model}.",
                        attempt=max_retries,
                        elapsed_sec=round(time.time() - started_at, 2),
                    )
                    return parsed
        log_llm_trace(
            run_id=run_id,
            env_name=env_name,
            stage=stage_label,
            trace_type="parse_error",
            llm_id=llm_id,
            provider=provider,
            model=model,
            base_url=base_url,
            goal=goal,
            request_payload=prompt,
            raw_response_text=output_text,
            parsed_response=body if isinstance(body, dict) else {},
            error=str(exc),
        )
        raise LlmProposalError(
            f"{stage_label} returned malformed JSON: {exc}",
            stage_label=stage_label,
            raw_response_text=output_text,
        ) from exc
    log_llm_trace(
        run_id=run_id,
        env_name=env_name,
        stage=stage_label,
        trace_type="response_received",
        llm_id=llm_id,
        provider=provider,
        model=model,
        base_url=base_url,
        goal=goal,
        request_payload=prompt,
        raw_response_text=output_text,
        parsed_response=parsed,
        details={"response_envelope": body if isinstance(body, dict) else {}},
    )
    update_task_config_request_status(
        run_id,
        status=f"{stage_label}_completed",
        message=f"{stage_label} completed successfully with model={model}.",
        attempt=max_retries,
        elapsed_sec=round(time.time() - started_at, 2),
    )
    return parsed


def call_llm_behavior_tag_plan(goal: str, env_name: str, *, run_id: str | None = None, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Takes in goal, environment name, and run ID and settings. 
    It updates the task configuration with the given information of
        run_id, status, message, model, base_url, env_name, goal, attempt, and elapsed_sec.
    Then it constructs a system prompt and user payload to send to the LLM.
    It then sends the prompt to the LLM
        LLM
    """
    settings = settings or _task_config_llm_settings()
    started_at = time.time()
    update_task_config_request_status(
        run_id,
        status="behavior_plan_preparing",
        message=f"Preparing behavior-tag planning request for model={settings['model']}.",
        model=settings["model"],
        base_url=settings["base_url"],
        env_name=env_name,
        goal=goal,
        attempt=0,
        elapsed_sec=0.0,
    )
    system_prompt = (
        "You are the first stage of a two-stage RL reward-design pipeline. "
        "Return valid JSON only. "
        "Map the user's goal to a small set of desired behavior tags and anti-goal tags using only the provided tag catalog. "
        "Assign each selected tag a weight between 0 and 1 representing importance. "
        "Prefer a concise set of high-signal tags over a long noisy list. "
        "Do not mention or invent observation or action variable names at this stage."
    )
    user_payload = {
        "env_name": env_name,
        "goal": goal,
        "available_behavior_tags": _compact_behavior_tag_catalog(env_name),
        "output_requirements": {
            "must_return_json_object": True,
            "required_keys": ["goal", "desired_tags", "avoid_tags", "constraints", "rationale"],
            "tag_item_fields": ["key", "weight", "reason"],
            "limits": {
                "max_desired_tags": 4,
                "max_avoid_tags": 3,
                "max_constraints": 3,
                "rationale_sentences_max": 2,
            },
        },
    }
    raw_plan = _call_llm_json_response(
        llm_id=settings.get("id"),
        provider=settings.get("provider"),
        base_url=settings["base_url"],
        model=settings["model"],
        api_key=settings["api_key"],
        timeout_sec=settings["timeout_sec"],
        max_retries=settings["max_retries"],
        system_prompt=system_prompt,
        user_payload=user_payload,
        run_id=run_id,
        env_name=env_name,
        goal=goal,
        stage_label="behavior_plan",
        max_output_tokens=settings["behavior_plan_max_tokens"],
    )
    plan = normalize_behavior_tag_plan(raw_plan, env_name, goal=goal)
    if not plan.get("desired_tags") and not plan.get("avoid_tags"):
        raise RuntimeError("behavior_plan response did not contain any valid behavior tags for this environment.")
    plan["provider"] = "glm" if "z.ai" in settings["base_url"] or "bigmodel" in settings["base_url"] else "openai-compatible"
    plan["model"] = settings["model"]
    update_task_config_request_status(
        run_id,
        status="behavior_plan_completed",
        message=f"Behavior-tag planning completed successfully with model={settings['model']}.",
        attempt=settings["max_retries"],
        elapsed_sec=round(time.time() - started_at, 2),
    )
    return plan


def call_llm_reward_config_from_behavior_plan(
    goal: str,
    env_name: str,
    behavior_plan: dict[str, Any],
    *,
    run_id: str | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = settings or _task_config_llm_settings()
    variable_specs = _compact_variable_specs_for_prompt(env_name)
    current_terms = _compact_reward_terms_for_prompt(env_name)
    system_prompt = (
        "You are the second stage of a two-stage RL reward-design pipeline. Return valid JSON only. "
        "Use the provided behavior plan to generate a structured RL task config with runnable reward expressions. "
        "Use only the provided formula variables and helper math functions already supported by the backend: "
        "abs, min, max, clip, sqrt, square, exp, log, sin, cos, tanh, sign. "
        "Do not invent variable names that are not in the provided variable list or in derived_signals that you also return. "
        "Do not invent previous-state variables such as prev_cart_position or prev_velocity unless they already appear exactly in the provided variable list. "
        "Use exact variable spellings only; if a variable is not listed, you must not reference it. "
        "Only these expression constructs are supported: names, numeric constants, +, -, *, /, %, **, unary +/- and helper function calls. "
        "Comparisons and control flow are not supported, including >, <, >=, <=, ==, !=, and ternary syntax like a ? b : c. "
        "If you need an indicator-like reward, approximate it using the supported math helpers instead of boolean logic. "
        "Derived signals are allowed, but each derived signal expression must itself be runnable using only the provided variables, task_params, "
        "and any earlier derived_signals in the returned list. Reward terms may reference returned derived_signals. "
        "The behavior plan is authoritative: reward the desired tags, penalize the avoid tags, and explain the mapping clearly. "
        "Keep the response compact: at most 2 task_params, 3 derived_signals, 5 reward_terms, 1 warning, and a rationale of at most 1 sentence. "
        "Keep every string field single-line plain text with no markdown, no code fences, and no embedded quotes unless required by valid JSON escaping. "
        "Keep descriptions very short. If a desired concept requires an unavailable variable, omit that term instead of inventing a new variable name."
    )
    compact_behavior_plan = {
        "goal": str(behavior_plan.get("goal") or goal or "").strip(),
        "desired_tags": list(behavior_plan.get("desired_tags") or [])[:4],
        "avoid_tags": list(behavior_plan.get("avoid_tags") or [])[:3],
        "constraints": list(behavior_plan.get("constraints") or [])[:3],
    }
    user_payload = {
        "env_name": env_name,
        "goal": goal,
        "behavior_plan": compact_behavior_plan,
        "available_variables": variable_specs,
        "default_reward_terms": current_terms,
        "formula_examples": reward_formula_examples_for_env(env_name),
        "output_requirements": {
            "must_return_json_object": True,
            "top_level_keys": ["goal", "task_params", "derived_signals", "reward_terms", "success_metric", "rationale", "warnings"],
            "derived_signal_fields": ["key", "expression", "description"],
            "reward_term_fields": ["key", "label", "description", "weight", "enabled", "expression"],
            "expression_rules": [
                "Use only provided variables, returned task_params, and earlier returned derived_signals.",
                "Do not use comparisons or ternary syntax.",
                "Do not use any variable name unless it appears exactly in available_variables or is introduced earlier as a derived signal.",
                "Do not invent prev_* state variables unless explicitly provided in available_variables.",
            ],
            "format_rules": [
                "All text values must be single-line strings.",
                "Keep descriptions under 8 words.",
                "Keep labels under 5 words.",
                "Do not include markdown, comments, or code fences.",
            ],
            "limits": {
                "max_task_params": 2,
                "max_derived_signals": 3,
                "max_reward_terms": 5,
                "max_warnings": 1,
                "rationale_sentences_max": 1,
            },
        },
    }
    proposal = _call_llm_json_response(
        llm_id=settings.get("id"),
        provider=settings.get("provider"),
        base_url=settings["base_url"],
        model=settings["model"],
        api_key=settings["api_key"],
        timeout_sec=settings["timeout_sec"],
        max_retries=settings["max_retries"],
        system_prompt=system_prompt,
        user_payload=user_payload,
        run_id=run_id,
        env_name=env_name,
        goal=goal,
        stage_label="reward_config",
        max_output_tokens=settings["reward_config_max_tokens"],
    )
    provider_label = "glm" if "z.ai" in settings["base_url"] or "bigmodel" in settings["base_url"] else "openai-compatible"
    proposal["provider"] = provider_label
    proposal["model"] = settings["model"]
    proposal["behavior_plan"] = behavior_plan
    return proposal


def call_llm_task_config_proposal(goal: str, env_name: str, *, run_id: str | None = None, llm_id: str | None = None) -> dict[str, Any]:
    settings = _task_config_llm_settings(llm_id)
    log_run_event(
        "task_config_llm_pipeline_started",
        run_id=run_id,
        env_name=env_name,
        details={
            "goal": goal,
            "llm_id": settings.get("id"),
            "provider": settings.get("provider"),
            "model": settings.get("model"),
            "base_url": settings.get("base_url"),
        },
    )
    try:
        behavior_plan = call_llm_behavior_tag_plan(goal, env_name, run_id=run_id, settings=settings)
    except Exception as exc:
        log_run_event(
            "task_config_behavior_plan_fallback",
            run_id=run_id,
            env_name=env_name,
            details={
                "goal": goal,
                "llm_id": settings.get("id"),
                "provider": settings.get("provider"),
                "model": settings.get("model"),
                "error": str(exc),
            },
        )
        behavior_plan = normalize_behavior_tag_plan(
            heuristic_behavior_tag_plan(goal, env_name),
            env_name,
            goal=goal,
        )
        behavior_plan.setdefault("warnings", [])
        behavior_plan["warnings"] = [f"Behavior-tag planning unavailable; used heuristic fallback instead. {exc}"]
        update_task_config_request_status(
            run_id,
            status="behavior_plan_fallback",
            message=behavior_plan["warnings"][0],
            elapsed_sec=task_config_request_status_by_run.get(run_id, {}).get("elapsed_sec", 0.0),
        )
    proposal = call_llm_reward_config_from_behavior_plan(
        goal,
        env_name,
        behavior_plan,
        run_id=run_id,
        settings=settings,
    )
    log_run_event(
        "task_config_llm_pipeline_completed",
        run_id=run_id,
        env_name=env_name,
        details={
            "goal": goal,
            "llm_id": settings.get("id"),
            "provider": settings.get("provider"),
            "model": settings.get("model"),
            "desired_tag_count": len(behavior_plan.get("desired_tags") or []),
            "avoid_tag_count": len(behavior_plan.get("avoid_tags") or []),
            "reward_term_count": len(proposal.get("reward_terms") or []),
        },
    )
    update_task_config_request_status(
        run_id,
        status="completed",
        message=f"Two-stage task-config proposal completed successfully with model={settings['model']}.",
        attempt=settings["max_retries"],
        elapsed_sec=task_config_request_status_by_run.get(run_id, {}).get("elapsed_sec", 0.0),
    )
    return proposal


def propose_task_config(goal: str, env_name: str, *, run_id: str | None = None, llm_id: str | None = None) -> dict[str, Any]:
    """
    Propose a task configuration based on the goal and environment name.
    Takes in a goal, environment name and two optional parameters: run_id and llm_id.
    Calls the LLM to get a reward configuration proposal based on the two-step reward design pipeline.
    If the LLM call fails, it falls back to a heuristic proposal based on the environment schema and goal keywords.
    It then normalizes and validates the proposed reward terms. If validation fails, it falls back to a heuristic proposal again.
    Returns the normalized reward configuration proposal.
    """
    try:
        proposal = call_llm_task_config_proposal(goal, env_name, run_id=run_id, llm_id=llm_id)
    except Exception as exc:
        proposal = build_heuristic_task_config_proposal(goal, env_name)
        proposal.setdefault("warnings", [])
        proposal["warnings"] = [f"Model proposal unavailable; used heuristic fallback instead. {exc}", *proposal["warnings"]]
        if isinstance(exc, LlmProposalError):
            proposal["raw_model_response"] = exc.raw_response_text
            proposal["model_proposal_preview"] = exc.partial_proposal
            proposal["llm_error_stage"] = exc.stage_label
        update_task_config_request_status(
            run_id,
            status="fallback",
            message=str(proposal["warnings"][0]),
            elapsed_sec=task_config_request_status_by_run.get(run_id, {}).get("elapsed_sec", 0.0),
        )
    proposal = normalize_task_config_proposal_shape(proposal)
    try:
        validate_task_config_for_run(proposal, env_name, run_id=None)
        proposal["reward_terms"] = normalize_reward_terms(
            env_name,
            proposal.get("reward_terms") or [],
        )
    except Exception as exc:
        fallback = build_heuristic_task_config_proposal(goal, env_name)
        fallback.setdefault("warnings", [])
        fallback["warnings"] = [f"Non-runnable proposal discarded; used heuristic fallback instead. {exc}", *fallback["warnings"]]
        fallback["model_proposal_preview"] = {
            key: value
            for key, value in proposal.items()
            if key not in {"warnings"}
        }
        if isinstance(proposal.get("_raw_model_response"), str):
            fallback["raw_model_response"] = proposal.get("_raw_model_response")
        proposal = fallback
        update_task_config_request_status(
            run_id,
            status="fallback_validation",
            message=str(fallback["warnings"][0]),
            elapsed_sec=task_config_request_status_by_run.get(run_id, {}).get("elapsed_sec", 0.0),
        )
    return proposal


def normalize_training_hyperparams(params: dict | None) -> dict:
    current = dict(DEFAULT_TRAINING_HYPERPARAMS)
    if params:
        current.update(params)

    current["learning_rate"] = float(current["learning_rate"])
    current["lr_schedule"] = str(current["lr_schedule"]).lower()
    if current["lr_schedule"] not in {"constant", "linear", "cosine"}:
        current["lr_schedule"] = "constant"

    current["n_steps"] = max(32, int(current["n_steps"]))
    current["batch_size"] = max(8, int(current["batch_size"]))
    current["n_epochs"] = max(1, int(current["n_epochs"]))
    current["gamma"] = float(current["gamma"])
    current["gae_lambda"] = float(current["gae_lambda"])
    current["clip_range"] = float(current["clip_range"])
    current["ent_coef"] = float(current["ent_coef"])
    current["vf_coef"] = float(current["vf_coef"])
    current["max_grad_norm"] = float(current["max_grad_norm"])
    model_size = str(current["model_size"]).lower()
    current["model_size"] = model_size if model_size in MODEL_SIZE_TO_NET_ARCH else "medium"
    return current


def get_training_hyperparams_for_run(run_id: str | None) -> dict:
    if not run_id:
        return dict(DEFAULT_TRAINING_HYPERPARAMS)
    if run_id not in training_hyperparams_by_run:
        training_hyperparams_by_run[run_id] = dict(DEFAULT_TRAINING_HYPERPARAMS)
    training_hyperparams_by_run[run_id] = normalize_training_hyperparams(training_hyperparams_by_run[run_id])
    return training_hyperparams_by_run[run_id]


def ensure_run_status(run_id: str | None, env_name: str | None = None) -> dict:
    if not run_id:
        return {}
    status = RUNS_TRAINING_STATUS.setdefault(run_id, {"status": "idle"})
    status.setdefault("env_name", env_name)
    status.setdefault("recent_training_episodes", [])
    status.setdefault("recent_rollout_episodes", [])
    status.setdefault("training_insights", {})
    status.setdefault("rollout_insights", {})
    status.setdefault("training_behavior_report", {})
    status.setdefault("rollout_behavior_report", {})
    status.setdefault("training_behavior_tags", {})
    status.setdefault("rollout_behavior_tags", {})
    if env_name:
        status["env_name"] = env_name
    return status


def append_recent_episode(run_id: str | None, episode_payload: dict[str, Any], source: str, env_name: str | None = None) -> None:
    status = ensure_run_status(run_id, env_name=env_name)
    if not status:
        return
    key = "recent_training_episodes" if source == "training" else "recent_rollout_episodes"
    insight_key = "training_insights" if source == "training" else "rollout_insights"
    behavior_key = "training_behavior_report" if source == "training" else "rollout_behavior_report"
    behavior_tag_key = "training_behavior_tags" if source == "training" else "rollout_behavior_tags"
    episodes = [episode_payload, *status.get(key, [])][:25]
    status[key] = episodes
    # Deterministic Insights section below
    status[insight_key] = build_episode_insights(
        episodes,
        source=source,
        env_name=status.get("env_name"),
    )
    # Behavior Metrics section below
    status[behavior_key] = build_behavior_metric_report(
        episodes,
        source=source,
        env_name=status.get("env_name"),
    )

    status[behavior_tag_key] = build_behavior_tag_report(
        status[behavior_key],
        status.get("env_name"),
    )


def build_learning_rate_schedule(initial_lr: float, strategy: str):
    strategy = strategy.lower()
    if strategy == "linear":
        return lambda progress_remaining: float(initial_lr) * max(float(progress_remaining), 0.0)
    if strategy == "cosine":
        return lambda progress_remaining: float(initial_lr) * 0.5 * (1.0 + np.cos(np.pi * (1.0 - max(float(progress_remaining), 0.0))))
    return float(initial_lr)


def get_reward_terms_for_run(run_id: str | None, env_name: str) -> list[dict]:
    if not run_id:
        return normalize_reward_terms(env_name, reward_template_for_env(env_name))

    current = reward_configs_by_run.get(run_id)
    if current is None or current.get("env_name") != env_name:
        reward_configs_by_run[run_id] = {
            "env_name": env_name,
            "terms": reward_template_for_env(env_name),
        }

    stored_terms = reward_configs_by_run[run_id]["terms"]
    normalized = validate_reward_terms(
        env_name,
        stored_terms,
        available_variable_names=task_variable_names_for_run(run_id, env_name),
    )
    reward_configs_by_run[run_id]["terms"] = normalized
    return normalized


def set_reward_terms_for_run(run_id: str, env_name: str, terms: list[dict]) -> list[dict]:
    normalized = validate_reward_terms(
        env_name,
        terms,
        available_variable_names=task_variable_names_for_run(run_id, env_name),
    )
    reward_configs_by_run[run_id] = {"env_name": env_name, "terms": normalized}
    log_reward_spec_snapshot(
        run_id=run_id,
        env_name=env_name,
        terms=normalized,
        task_config=get_task_config_for_run(run_id, env_name),
        source="set_reward_terms_for_run",
    )
    return normalized


def infer_reward_config_source_type(run_id: str | None, env_name: str) -> str:
    task_config = get_task_config_for_run(run_id, env_name)
    provider = str(task_config.get("provider") or "").strip().lower()
    if provider == "heuristic":
        return "heuristic"
    if provider and provider not in {"none", "manual"}:
        return "llm"
    return "manual"


def list_saved_reward_config_files(env_name: str | None = None) -> list[str]:
    files = [path.name for path in REWARD_CONFIGS_DIR.iterdir() if path.is_file() and path.suffix == ".json"]
    if not env_name:
        return sorted(files)
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", env_name.strip()) if env_name.strip() else ""
    filtered = [name for name in files if name.startswith(f"{prefix}__") or f"__{prefix}__" in name or name.endswith(f"__{prefix}.json")]
    return sorted(filtered)


def build_reward_config_payload(run_id: str, env_name: str, *, source_type: str | None = None) -> dict[str, Any]:
    task_config = get_task_config_for_run(run_id, env_name)
    inferred_source_type = (source_type or infer_reward_config_source_type(run_id, env_name)).strip().lower() or "manual"
    terms = get_reward_terms_for_run(run_id, env_name)
    readable_terms = []
    for term in terms:
        label = str(term.get("label") or term.get("key") or "Reward Term").strip()
        weight = float(term.get("weight", 0.0))
        expression = str(term.get("expression") or "").strip() or "native_reward"
        description = str(term.get("description") or "").strip()
        state_text = "enabled" if bool(term.get("enabled", True)) else "disabled"
        paragraph = f"{label} (w={weight:.2f}, {state_text}) uses `{expression}`."
        if description:
            paragraph = f"{paragraph} {description}"
        readable_terms.append(paragraph)
    return {
        "run_id": run_id,
        "env_name": env_name,
        "saved_at": datetime.now().isoformat(),
        "source_type": inferred_source_type,
        "terms": terms,
        "terms_summary": "\n\n".join(readable_terms),
        "task_config": task_config,
    }


def save_reward_config_to_disk(run_id: str, env_name: str, *, filename: str | None = None, source_type: str | None = None) -> tuple[str, dict[str, Any]]:
    safe_env_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", env_name.strip() or "env")
    inferred_source_type = (source_type or infer_reward_config_source_type(run_id, env_name)).strip().lower() or "manual"
    default_name = f"{safe_env_name}__{inferred_source_type}__reward_config.json"
    resolved_name = sanitize_storage_name(filename or default_name, ".json")
    payload = build_reward_config_payload(run_id, env_name, source_type=inferred_source_type)
    with open(REWARD_CONFIGS_DIR / resolved_name, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return resolved_name, payload


def load_reward_config_from_disk(filename: str) -> dict[str, Any]:
    safe_name = sanitize_storage_name(filename, ".json")
    path = (REWARD_CONFIGS_DIR / safe_name).resolve()
    try:
        path.relative_to(REWARD_CONFIGS_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="reward config path must stay inside the managed reward_configs directory") from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Saved reward config '{safe_name}' not found.")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Saved reward config file is invalid.")
    return payload


def reward_weights_for_run(run_id: str | None, env_name: str) -> dict[str, float]:
    return {
        term["key"]: float(term["weight"]) if term["enabled"] else 0.0
        for term in get_reward_terms_for_run(run_id, env_name)
    }


def make_reward_wrapped_env(
    env_name: str,
    run_id: str | None,
    render_mode: str | None = None,
    reward_terms_override: list[dict] | None = None,
    task_config_override: dict[str, Any] | None = None,
):
    env_kwargs = {"render_mode": render_mode} if render_mode else {}
    env = gym.make(env_name, **env_kwargs)
    return RewardShapingWrapper(
        env,
        env_name=env_name,
        config_provider=lambda: reward_terms_override if reward_terms_override is not None else get_reward_terms_for_run(run_id, env_name),
        task_config_provider=lambda: task_config_override if task_config_override is not None else get_task_config_for_run(run_id, env_name),
    )


def make_training_env_factory(env_name: str, run_id: str | None):
    ensure_sb3_available()
    current_terms = get_reward_terms_for_run(run_id, env_name)
    info_keywords = tuple(
        [f"reward_{term['key']}" for term in current_terms] + ["reward_total"]
    )

    def _factory():
        env = make_reward_wrapped_env(env_name, run_id=run_id, render_mode=None)
        return Monitor(env, info_keywords=info_keywords)

    return _factory


def build_reward_ablation_configs(env_name: str, terms: list[dict]) -> list[dict[str, Any]]:
    normalized_terms = normalize_reward_terms(env_name, terms)
    native_term = next((term for term in normalized_terms if term["key"] == "native"), None)
    if native_term is None:
        native_term = {
            "key": "native",
            "label": "Native Reward",
            "description": "Original environment reward.",
            "weight": 1.0,
            "enabled": True,
            "expression": "",
            "is_custom": False,
        }

    active_terms = [term for term in normalized_terms if term.get("enabled", True)]
    extra_terms = [term for term in active_terms if term["key"] != "native"]
    configs: list[dict[str, Any]] = []

    def _native_only_terms() -> list[dict]:
        return [dict(native_term, enabled=True, weight=float(native_term.get("weight", 1.0)))]

    configs.append({
        "label": "Native only",
        "terms": _native_only_terms(),
        "term_keys": ["native"],
    })

    for term in extra_terms:
        configs.append({
            "label": f"Native + {term['label']}",
            "terms": _native_only_terms() + [dict(term)],
            "term_keys": ["native", term["key"]],
        })

    configs.append({
        "label": "All enabled terms",
        "terms": [dict(term) for term in active_terms] or _native_only_terms(),
        "term_keys": [term["key"] for term in active_terms] or ["native"],
    })
    return configs


def collect_policy_eval_trajectories(model: Any, env_name: str, episodes: int = 3) -> list[dict[str, Any]]:
    eval_env = make_reward_wrapped_env(
        env_name,
        run_id=None,
        render_mode=None,
        reward_terms_override=[{
            "key": "native",
            "label": "Native Reward",
            "description": "Original environment reward.",
            "weight": 1.0,
            "enabled": True,
            "expression": "",
            "is_custom": False,
        }],
    )
    trajectories: list[dict[str, Any]] = []
    try:
        for _ in range(max(1, episodes)):
            obs, _ = eval_env.reset()
            done = False
            native_total = 0.0
            steps = 0
            reward_history: list[dict[str, Any]] = []
            while not done and steps < 2000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                raw_terms = {key: float(value) for key, value in info.get("reward_raw_terms", {}).items()}
                reward_history.append({
                    "step": steps + 1,
                    "reward_raw_terms": raw_terms,
                })
                native_total += float(raw_terms.get("native", reward))
                done = bool(terminated or truncated)
                steps += 1
            trajectories.append({
                "reward_history": reward_history,
                "native_total": native_total,
            })
    finally:
        eval_env.close()

    return trajectories


def rescore_trajectories_with_reward_terms(trajectories: list[dict[str, Any]], terms: list[dict]) -> dict[str, float]:
    normalized_terms = [
        {
            "key": term["key"],
            "weight": float(term.get("weight", 0.0)),
            "enabled": bool(term.get("enabled", True)),
        }
        for term in terms
    ]
    shaped_returns: list[float] = []
    native_returns: list[float] = []

    for trajectory in trajectories:
        shaped_total = 0.0
        native_total = float(trajectory.get("native_total", 0.0))
        for step_entry in trajectory.get("reward_history", []):
            raw_terms = step_entry.get("reward_raw_terms", {}) or {}
            for term in normalized_terms:
                raw_value = float(raw_terms.get(term["key"], 0.0))
                if term.get("enabled", True):
                    shaped_total += float(term.get("weight", 0.0)) * raw_value
        shaped_returns.append(shaped_total)
        native_returns.append(native_total)

    return {
        "avg_eval_reward": float(np.mean(shaped_returns)) if shaped_returns else 0.0,
        "avg_native_reward": float(np.mean(native_returns)) if native_returns else 0.0,
    }


#@app.post("/start")
def start_training(run_id: str = None, env_name: str = "CartPole-v1", train_steps:int = 1000, reset_num_timesteps = False, callback: Any = None,
                   ws_manager=None, frame_fn=None, every_n_steps:int = 100, model: Any = None):
    ensure_sb3_available()
    ensure_run_status(run_id, env_name=env_name)
    RUNS_TRAINING_STATUS[run_id].setdefault("model_path", None)
    RUNS_TRAINING_STATUS[run_id].setdefault("error", None)
    RUNS_TRAINING_STATUS[run_id]["status"] = "running"
    RUNS_TRAINING_STATUS[run_id]["reward_ablation"] = None
    RUNS_TRAINING_STATUS[run_id]["reward_ablation_status"] = "idle"
    preview_model = None
    eval_env = None
    training_hyperparams = get_training_hyperparams_for_run(run_id)
    current_reward_terms = get_reward_terms_for_run(run_id, env_name)
    current_task_config = get_task_config_for_run(run_id, env_name)
    log_run_event(
        "training_start_requested",
        run_id=run_id,
        env_name=env_name,
        details={
            "train_steps": int(train_steps),
            "reset_num_timesteps": bool(reset_num_timesteps),
            "hyperparams": training_hyperparams,
            "model_path_requested": trained_model_paths.get(run_id),
            "device": training_model_devices.get(run_id),
        },
    )
    log_reward_spec_snapshot(
        run_id=run_id,
        env_name=env_name,
        terms=current_reward_terms,
        task_config=current_task_config,
        source="training_start",
        details={"train_steps": int(train_steps)},
    )

    if model is None:
        vec_env = DummyVecEnv(
            [make_training_env_factory(env_name, run_id=run_id) for _ in range(8)]
        )

        device = "cpu"
        if run_id in training_model_devices:
            device = training_model_devices[run_id]

        policy_kwargs = {
            "net_arch": dict(
                pi=MODEL_SIZE_TO_NET_ARCH[training_hyperparams["model_size"]],
                vf=MODEL_SIZE_TO_NET_ARCH[training_hyperparams["model_size"]],
            )
        }

        learning_rate = build_learning_rate_schedule(
            training_hyperparams["learning_rate"],
            training_hyperparams["lr_schedule"],
        )

        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device=device,
            tensorboard_log="./tensorboard_logs",
            learning_rate=learning_rate,
            n_steps=training_hyperparams["n_steps"],
            batch_size=training_hyperparams["batch_size"],
            n_epochs=training_hyperparams["n_epochs"],
            gamma=training_hyperparams["gamma"],
            gae_lambda=training_hyperparams["gae_lambda"],
            clip_range=training_hyperparams["clip_range"],
            ent_coef=training_hyperparams["ent_coef"],
            vf_coef=training_hyperparams["vf_coef"],
            max_grad_norm=training_hyperparams["max_grad_norm"],
            policy_kwargs=policy_kwargs,
        )

        preview_env = DummyVecEnv(
            [make_training_env_factory(env_name, run_id=run_id)]
        )
        preview_model = PPO(
            "MlpPolicy",
            preview_env,
            verbose=0,
            device="cpu",
            learning_rate=learning_rate,
            n_steps=training_hyperparams["n_steps"],
            batch_size=training_hyperparams["batch_size"],
            n_epochs=training_hyperparams["n_epochs"],
            gamma=training_hyperparams["gamma"],
            gae_lambda=training_hyperparams["gae_lambda"],
            clip_range=training_hyperparams["clip_range"],
            ent_coef=training_hyperparams["ent_coef"],
            vf_coef=training_hyperparams["vf_coef"],
            max_grad_norm=training_hyperparams["max_grad_norm"],
            policy_kwargs=policy_kwargs,
        )
        preview_model.policy.load_state_dict(model.policy.state_dict())
        training_preview_model[run_id] = preview_model
        eval_env = make_reward_wrapped_env(env_name, run_id=run_id, render_mode=None)

    class ModelSyncCallback(BaseCallback):
        def __init__(self, run_id: str, source_model: Any, target_model: Any, every_n_steps: int = 100, eval_env=None, status_dict=None):
            super().__init__()
            self.run_id = run_id
            self.source_model = source_model
            self.target_model = target_model
            self.every_n_steps = max(1, every_n_steps)
            self.eval_every_n_steps = max(1000, every_n_steps * 5)
            self.eval_env = eval_env
            self.status_dict = status_dict
            self._last_sync = 0
            self._last_eval = 0

        def _sync_once(self):
            if self.target_model is None:
                return
            with training_preview_model_locks[self.run_id]:
                self.target_model.policy.load_state_dict(self.source_model.policy.state_dict())

        def _evaluate_once(self):
            if self.target_model is None or self.eval_env is None or self.status_dict is None:
                return
            totals: list[float] = []
            for _ in range(3):
                obs, _ = self.eval_env.reset()
                total_reward = 0.0
                for _ in range(1000):
                    with training_preview_model_locks[self.run_id]:
                        action, _ = self.target_model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    total_reward += float(reward)
                    if terminated or truncated:
                        break
                totals.append(total_reward)
            self.status_dict[self.run_id]["eval_reward"] = float(np.mean(totals)) if totals else 0.0

        def _on_step(self) -> bool:
            if self.target_model is None:
                return True
            steps_done = int(getattr(self.model, "num_timesteps", 0)) # type: ignore[attr-defined]
            if (steps_done - self._last_sync) >= self.every_n_steps:
                self._last_sync = steps_done
                self._sync_once()
            if (steps_done - self._last_eval) >= self.eval_every_n_steps:
                self._last_eval = steps_done
                self._evaluate_once()
            return True

        def _on_training_end(self) -> None:
            self._sync_once()
            self._evaluate_once()

    # Build a progress callback
    progress_data_cb = TrainingProgressCallback(
        run_id=run_id,
        status_dict=RUNS_TRAINING_STATUS,
        total_steps=train_steps,
        every_n_steps=every_n_steps,
        frame_fn=frame_fn,                                  # None if you don't want frames
        send_tick=make_tick_sender(ws_manager) if ws_manager else None,
        send_frame=None#make_frame_sender(ws_manager, encoder=encode_jpeg) if (ws_manager and frame_fn) else None,
    )

    sync_model_cb = ModelSyncCallback(
        run_id,
        model,
        preview_model,
        every_n_steps=every_n_steps,
        eval_env=eval_env,
        status_dict=RUNS_TRAINING_STATUS,
    )
    callback_items = [progress_data_cb, sync_model_cb]
    if callback is not None:
        callback_items.append(callback)
    callbacks_list = CallbackList(callback_items)

    def train():
        global train_run_progress, RUNS_TRAINING_STATUS
        train_run_progress = 0
        try:
            RUNS_TRAINING_STATUS[run_id]["status"] = "running"
            RUNS_TRAINING_STATUS[run_id]["error"] = None
            RUNS_TRAINING_STATUS[run_id]["hyperparams"] = training_hyperparams
            model.learn(total_timesteps=train_steps, reset_num_timesteps=reset_num_timesteps, callback=callbacks_list)
            stop_requested = bool(RUNS_TRAINING_STATUS[run_id].get("stop", False))
            RUNS_TRAINING_STATUS[run_id]["reward_ablation_status"] = "running"
            try:
                ablation_terms = get_reward_terms_for_run(run_id, env_name)
                ablation_configs = build_reward_ablation_configs(env_name, ablation_terms)
                ablation_trajectories = collect_policy_eval_trajectories(model, env_name, episodes=3)
                ablation_rows = []
                for config in ablation_configs:
                    results = rescore_trajectories_with_reward_terms(ablation_trajectories, config["terms"])
                    ablation_rows.append({
                        "label": config["label"],
                        "term_keys": config["term_keys"],
                        **results,
                    })
                RUNS_TRAINING_STATUS[run_id]["reward_ablation"] = {
                    "env_name": env_name,
                    "eval_episodes": len(ablation_trajectories),
                    "trajectory_mode": "fixed_counterfactual_rescoring",
                    "rows": ablation_rows,
                }
                RUNS_TRAINING_STATUS[run_id]["reward_ablation_status"] = "done"
            except Exception as exc:
                RUNS_TRAINING_STATUS[run_id]["reward_ablation"] = {
                    "env_name": env_name,
                    "error": str(exc),
                    "rows": [],
                }
                RUNS_TRAINING_STATUS[run_id]["reward_ablation_status"] = "error"
            train_model_actual_path = ""
            if run_id not in trained_model_paths:
                train_model_actual_path = build_default_model_path(env_name)
            else:
                train_model_actual_path = trained_model_paths[run_id]
            model.save(train_model_actual_path)
            write_model_metadata(
                Path(train_model_actual_path),
                env_name=env_name,
                run_id=run_id,
                source="training",
                details={"train_steps": int(train_steps)},
            )
            RUNS_TRAINING_STATUS[run_id]["model_path"] = train_model_actual_path
            RUNS_TRAINING_STATUS[run_id]["status"] = "stopped" if stop_requested else "done"
            log_run_event(
                "training_finished",
                run_id=run_id,
                env_name=env_name,
                details={
                    "status": RUNS_TRAINING_STATUS[run_id]["status"],
                    "model_path": train_model_actual_path,
                    "stop_requested": stop_requested,
                    "steps_done": RUNS_TRAINING_STATUS[run_id].get("steps_done"),
                    "reward_last": RUNS_TRAINING_STATUS[run_id].get("reward_last"),
                    "reward_mean": RUNS_TRAINING_STATUS[run_id].get("reward_mean"),
                    "eval_reward": RUNS_TRAINING_STATUS[run_id].get("eval_reward"),
                },
            )
            print("Training stopped early" if stop_requested else "Training complete")
        except Exception as e:
            RUNS_TRAINING_STATUS[run_id]["status"] = "error"
            RUNS_TRAINING_STATUS[run_id]["error"] = str(e)
            logger.exception("Training failed for run_id=%s env_name=%s", run_id, env_name)
            log_run_event(
                "training_failed",
                run_id=run_id,
                env_name=env_name,
                details={"error": str(e)},
            )
        finally:
            RUNS_TRAINING_STATUS[run_id]["stop"] = False
            model_env = model.get_env()
            if model_env is not None:
                model_env.close()
            if eval_env is not None:
                eval_env.close()
            if run_id in training_preview_model:
                with training_preview_model_locks[run_id]:
                    training_preview_model[run_id] = preview_model
    Thread(target=train).start()
    # status option
    return model

from fastapi import UploadFile, File
print("Models directory: ", MODELS_DIR)
@app.get("/models")
def list_models():
    records = list_model_records()
    records.sort(key=lambda entry: (entry["created_ts"], entry["name"]))
    return {
        "models": [entry["name"] for entry in records],
        "model_records": records,
    }

class RolloutSpeedRequest(BaseModel):
    run_id: str
    fps: int
    delay: float
@app.post("/rollout_speed")
def change_rollout_speed(rollReq:RolloutSpeedRequest):
    global time_intervals, rollout_fps
    runId = rollReq.run_id
    rollout_fps[runId] = rollReq.fps
    time_intervals[runId] = rollReq.delay
    print(f"Run id is {runId}. Time interval is : {rollReq.delay}")
    return {"fps": rollout_fps[runId], "delay": time_intervals[runId]}


@app.get("/training_runs/{run_id}")
def get_run(run_id: str):
    status = RUNS_TRAINING_STATUS.get(run_id)
    if status is None:
        return {"status": "unknown"}
    # Getting recent_training_episodes and training_insights. If no training insights, build it. 
    if status.get("recent_training_episodes") and not status.get("training_insights"):
        status["training_insights"] = build_episode_insights(
            status.get("recent_training_episodes", []),
            source="training",
            env_name=status.get("env_name"),
        )
    if status.get("recent_training_episodes") and not status.get("training_behavior_report"):
        status["training_behavior_report"] = build_behavior_metric_report(
            status.get("recent_training_episodes", []),
            source="training",
            env_name=status.get("env_name"),
        )
    if status.get("training_behavior_report") and not status.get("training_behavior_tags"):
        status["training_behavior_tags"] = build_behavior_tag_report(
            status.get("training_behavior_report", {}),
            status.get("env_name"),
        )
    if status.get("recent_rollout_episodes") and not status.get("rollout_insights"):
        status["rollout_insights"] = build_episode_insights(
            status.get("recent_rollout_episodes", []),
            source="rollout",
            env_name=status.get("env_name"),
        )
    if status.get("recent_rollout_episodes") and not status.get("rollout_behavior_report"):
        status["rollout_behavior_report"] = build_behavior_metric_report(
            status.get("recent_rollout_episodes", []),
            source="rollout",
            env_name=status.get("env_name"),
        )
    if status.get("rollout_behavior_report") and not status.get("rollout_behavior_tags"):
        status["rollout_behavior_tags"] = build_behavior_tag_report(
            status.get("rollout_behavior_report", {}),
            status.get("env_name"),
        )
    return status


@app.get("/reward_config")
def get_reward_config(run_id: str, env_name: str):
    terms = get_reward_terms_for_run(run_id, env_name)
    task_config = get_task_config_for_run(run_id, env_name)
    llm_catalog = _task_config_llm_catalog()
    default_settings = None
    try:
        default_settings = _task_config_llm_settings()
    except RuntimeError:
        default_settings = None
    return {
        "run_id": run_id,
        "env_name": env_name,
        "terms": terms,
        "supports_custom_reward": len(terms) > 1,
        "available_variables": task_variable_specs_for_run(run_id, env_name),
        "formula_examples": reward_formula_examples_for_env(env_name),
        "task_config": task_config,
        "source_type": infer_reward_config_source_type(run_id, env_name),
        "available_behavior_tags": available_behavior_tags_for_env(env_name),
        "available_llms": [{key: value for key, value in item.items() if key not in {"api_key"}} for item in llm_catalog],
        "default_llm_id": default_settings.get("id") if default_settings else None,
        "saved_reward_configs": list_saved_reward_config_files(env_name),
        "reward_source_links": [
            {
                "label": "Open Reward Templates",
                "path": str((BASE_DIR / "train_backend_reward_tuning" / "reward_templates.py").resolve()),
            },
            {
                "label": "Open Reward Shaping Wrapper",
                "path": str((BASE_DIR / "train_backend_reward_tuning" / "reward_shaping.py").resolve()),
            },
        ],
    }


@app.post("/reward_config")
def update_reward_config(req: RewardConfigUpdateRequest):
    llm_catalog = _task_config_llm_catalog()
    default_settings = None
    try:
        default_settings = _task_config_llm_settings()
    except RuntimeError:
        default_settings = None
    try:
        has_task_overrides = bool(
            (req.goal and str(req.goal).strip())
            or req.task_params
            or req.derived_signals
            or req.success_metric
            or req.rationale
            or req.warnings
            or req.behavior_plan
        )
        if has_task_overrides:
            proposed_config = {
                "goal": req.goal or "",
                "task_params": req.task_params,
                "derived_signals": req.derived_signals,
                "reward_terms": [term.model_dump() for term in req.terms],
                "success_metric": req.success_metric or "",
                "rationale": req.rationale or "",
                "warnings": req.warnings,
                "provider": req.provider or "manual",
                "model": req.model or "",
                "behavior_plan": req.behavior_plan or {},
            }
            proposed_config = normalize_task_config_proposal_shape(proposed_config)
            validate_task_config_for_run(proposed_config, req.env_name, run_id=req.run_id)
            set_task_config_for_run(req.run_id, req.env_name, proposed_config)
        terms = set_reward_terms_for_run(
            run_id=req.run_id,
            env_name=req.env_name,
            terms=[term.model_dump() for term in req.terms],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "updated",
        "run_id": req.run_id,
        "env_name": req.env_name,
        "terms": terms,
        "available_variables": task_variable_specs_for_run(req.run_id, req.env_name),
        "formula_examples": reward_formula_examples_for_env(req.env_name),
        "task_config": get_task_config_for_run(req.run_id, req.env_name),
        "source_type": infer_reward_config_source_type(req.run_id, req.env_name),
        "available_behavior_tags": available_behavior_tags_for_env(req.env_name),
        "available_llms": [{key: value for key, value in item.items() if key not in {"api_key"}} for item in llm_catalog],
        "default_llm_id": default_settings.get("id") if default_settings else None,
        "saved_reward_configs": list_saved_reward_config_files(req.env_name),
        "reward_source_links": [
            {
                "label": "Open Reward Templates",
                "path": str((BASE_DIR / "train_backend_reward_tuning" / "reward_templates.py").resolve()),
            },
            {
                "label": "Open Reward Shaping Wrapper",
                "path": str((BASE_DIR / "train_backend_reward_tuning" / "reward_shaping.py").resolve()),
            },
        ],
    }


@app.post("/propose_task_config")
def propose_task_config_endpoint(req: TaskConfigProposalRequest):
    update_task_config_request_status(
        req.run_id,
        status="queued",
        message="Task-config proposal request queued.",
        attempt=0,
        elapsed_sec=0.0,
        env_name=req.env_name,
        goal=req.goal,
        llm_id=req.llm_id,
    )
    proposal = propose_task_config(req.goal, req.env_name, run_id=req.run_id, llm_id=req.llm_id)
    return {
        "run_id": req.run_id,
        "env_name": req.env_name,
        "available_behavior_tags": available_behavior_tags_for_env(req.env_name),
        # collect all information OTHER than the API_key for available LLMs to return in the response. 
        "available_llms": [
            {key: value for key, value in item.items() if key not in {"api_key"}}
            for item in _task_config_llm_catalog()
        ],
        **proposal,
    }


@app.get("/task_config_llms")
def get_task_config_llms():
    catalog = _task_config_llm_catalog()
    default_settings = None
    try:
        default_settings = _task_config_llm_settings()
    except RuntimeError:
        default_settings = None
    return {
        "llms": [{key: value for key, value in item.items() if key not in {"api_key"}} for item in catalog],
        "default_llm_id": default_settings.get("id") if default_settings else None,
    }


@app.get("/task_config_status/{run_id}")
def get_task_config_status(run_id: str):
    return task_config_request_status_by_run.get(
        run_id,
        {
            "run_id": run_id,
            "status": "idle",
            "message": "No active proposal request.",
            "attempt": 0,
            "elapsed_sec": 0.0,
        },
    )


@app.post("/apply_task_config")
def apply_task_config(req: TaskConfigApplyRequest):
    llm_catalog = _task_config_llm_catalog()
    default_settings = None
    try:
        default_settings = _task_config_llm_settings()
    except RuntimeError:
        default_settings = None
    proposed_config = {
        "goal": req.goal or "",
        "task_params": req.task_params,
        "derived_signals": req.derived_signals,
        "reward_terms": [term.model_dump() for term in req.reward_terms],
        "success_metric": req.success_metric or "",
        "rationale": req.rationale or "",
        "warnings": req.warnings,
        "provider": req.provider or "manual",
        "model": req.model or "",
        "behavior_plan": req.behavior_plan or {},
    }
    try:
        validate_task_config_for_run(proposed_config, req.env_name, run_id=req.run_id)
        task_config = set_task_config_for_run(
            req.run_id,
            req.env_name,
            proposed_config,
        )
        terms = set_reward_terms_for_run(
            run_id=req.run_id,
            env_name=req.env_name,
            terms=proposed_config["reward_terms"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "applied",
        "run_id": req.run_id,
        "env_name": req.env_name,
        "terms": terms,
        "task_config": task_config,
        "available_variables": task_variable_specs_for_run(req.run_id, req.env_name),
        "formula_examples": reward_formula_examples_for_env(req.env_name),
        "source_type": infer_reward_config_source_type(req.run_id, req.env_name),
        "available_behavior_tags": available_behavior_tags_for_env(req.env_name),
        "available_llms": [{key: value for key, value in item.items() if key not in {"api_key"}} for item in llm_catalog],
        "default_llm_id": default_settings.get("id") if default_settings else None,
        "saved_reward_configs": list_saved_reward_config_files(req.env_name),
        "reward_source_links": [
            {
                "label": "Open Reward Templates",
                "path": str((BASE_DIR / "train_backend_reward_tuning" / "reward_templates.py").resolve()),
            },
            {
                "label": "Open Reward Shaping Wrapper",
                "path": str((BASE_DIR / "train_backend_reward_tuning" / "reward_shaping.py").resolve()),
            },
        ],
    }


@app.get("/reward_config_files")
def get_reward_config_files(env_name: str | None = None):
    return {"reward_configs": list_saved_reward_config_files(env_name)}


@app.post("/save_reward_config")
def save_reward_config(req: SaveRewardConfigRequest):
    filename, payload = save_reward_config_to_disk(
        req.run_id,
        req.env_name,
        filename=req.filename,
        source_type=req.source_type,
    )
    return {
        "status": "saved",
        "filename": filename,
        "source_type": payload.get("source_type", "manual"),
        "saved_reward_configs": list_saved_reward_config_files(req.env_name),
        "task_config": payload.get("task_config", {}),
    }


@app.post("/load_reward_config")
def load_reward_config(req: LoadRewardConfigRequest):
    llm_catalog = _task_config_llm_catalog()
    default_settings = None
    try:
        default_settings = _task_config_llm_settings()
    except RuntimeError:
        default_settings = None
    payload = load_reward_config_from_disk(req.filename)
    file_env_name = str(payload.get("env_name") or req.env_name).strip() or req.env_name
    if file_env_name != req.env_name:
        raise HTTPException(
            status_code=400,
            detail=f"Saved reward config targets env '{file_env_name}', not '{req.env_name}'.",
        )
    task_config = payload.get("task_config") or {}
    terms = payload.get("terms") or []
    proposed_config = {
        "goal": str(task_config.get("goal") or ""),
        "task_params": list(task_config.get("task_params") or []),
        "derived_signals": list(task_config.get("derived_signals") or []),
        "reward_terms": terms,
        "success_metric": str(task_config.get("success_metric") or ""),
        "rationale": str(task_config.get("rationale") or ""),
        "warnings": list(task_config.get("warnings") or []),
        "provider": str(task_config.get("provider") or payload.get("source_type") or "manual"),
        "model": str(task_config.get("model") or ""),
        "behavior_plan": dict(task_config.get("behavior_plan") or {}),
    }
    try:
        validate_task_config_for_run(proposed_config, req.env_name, run_id=req.run_id)
        stored_task_config = set_task_config_for_run(req.run_id, req.env_name, proposed_config)
        stored_terms = set_reward_terms_for_run(req.run_id, req.env_name, terms)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "loaded",
        "filename": sanitize_storage_name(req.filename, ".json"),
        "run_id": req.run_id,
        "env_name": req.env_name,
        "terms": stored_terms,
        "task_config": stored_task_config,
        "available_variables": task_variable_specs_for_run(req.run_id, req.env_name),
        "formula_examples": reward_formula_examples_for_env(req.env_name),
        "source_type": str(payload.get("source_type") or infer_reward_config_source_type(req.run_id, req.env_name)),
        "available_behavior_tags": available_behavior_tags_for_env(req.env_name),
        "available_llms": [{key: value for key, value in item.items() if key not in {"api_key"}} for item in llm_catalog],
        "default_llm_id": default_settings.get("id") if default_settings else None,
        "saved_reward_configs": list_saved_reward_config_files(req.env_name),
        "reward_source_links": [
            {
                "label": "Open Reward Templates",
                "path": str((BASE_DIR / "train_backend_reward_tuning" / "reward_templates.py").resolve()),
            },
            {
                "label": "Open Reward Shaping Wrapper",
                "path": str((BASE_DIR / "train_backend_reward_tuning" / "reward_shaping.py").resolve()),
            },
        ],
    }



class PauseRequest(BaseModel):
    paused: bool
class StopTrainingRequest(BaseModel):
    run_id: str
@app.get("/progress/{run_id}")
def get_progress(run_id: str):
    status = RUNS_TRAINING_STATUS.get(run_id, {})
    steps_done = int(status.get("steps_done", 0) or 0)
    total_steps = int(status.get("total_steps", 0) or 0)
    if total_steps > 0:
        progress = 100 * min(steps_done, total_steps) / total_steps
    else:
        progress = train_run_progresses.get(run_id, 0)
    if status.get("status") == "done":
        progress = 100
    progress_bar_log_file.write(f"Put variable by name progress {progress} for run_id {run_id}\n")
    progress_bar_log_file.flush()
    return {
        "progress": progress,
        "status": status.get("status", "unknown"),
        "steps_done": steps_done,
        "total_steps": total_steps,
        "error": status.get("error"),
    }

@app.post("/stop_training")
def stop_training(req: StopTrainingRequest):
    run_id = req.run_id
    status = RUNS_TRAINING_STATUS.setdefault(run_id, {"status": "unknown"})
    status["stop"] = True
    if status.get("status") == "running":
        status["status"] = "stopping"
    return {"status": status.get("status", "stopping"), "run_id": run_id}

import uuid
def generate_run_id():
    return str(uuid.uuid4())  # e.g., '550e8400-e29b-41d4-a716-446655440000'
def generate_session_id():
    return str(uuid.uuid4())  # e.g., '550e8400-e29b-41d4-a716-446655440000'

@app.get("/unique_run_id")
def get_unique_run_id():
    run_id = generate_run_id()
    print("run_id:", run_id)
    return {"run_id": run_id}

rollout_pause_state = {}# run_id: paused or not
from fastapi import Body
from pydantic import BaseModel
class PauseRequest(BaseModel):
    session_id: str
    paused: bool
@app.post("/pause_rollout")
def pause_rollout(req: PauseRequest):
    session_id = req.session_id
    rollout_pause_state[session_id] = req.paused
    return {"status": "paused" if req.paused else "resumed",  
            "session_id": session_id}

class SetTrainingDirRequest(BaseModel):
    run_id: str
    train_dir_path: str
    device:str
    env_name: str | None = None
    training_hyperparams: dict | None = None

@app.post("/set_training_dir")
def set_training_dir(req: SetTrainingDirRequest):
    run_id = req.run_id
    where_to_save_trained_model = resolve_model_output_path(req.train_dir_path, run_id, req.env_name or env_name or "env")

    print("New parameter obtained: Here is where save trained model - ", where_to_save_trained_model)
    device = req.device
    print("Device obtainied ", device)
    trained_model_paths[run_id] = str(where_to_save_trained_model)
    training_model_devices[run_id] = device
    training_hyperparams_by_run[run_id] = normalize_training_hyperparams(req.training_hyperparams)
    log_run_event(
        "training_dir_set",
        run_id=run_id,
        env_name=req.env_name or env_name or "env",
        details={
            "path": str(where_to_save_trained_model),
            "device": device,
            "training_hyperparams": training_hyperparams_by_run[run_id],
        },
    )
    log_reward_spec_snapshot(
        run_id=run_id,
        env_name=req.env_name or env_name or "env",
        terms=get_reward_terms_for_run(run_id, req.env_name or env_name or "env"),
        task_config=get_task_config_for_run(run_id, req.env_name or env_name or "env"),
        source="set_training_dir",
        details={"path": str(where_to_save_trained_model)},
    )
    # Here you would typically set the training directory for the session
    return {
        "status": "training directory set",
        "run_id": run_id,
        "path": str(where_to_save_trained_model),
        "training_hyperparams": training_hyperparams_by_run[run_id],
    }

class SaveRolloutRequest(BaseModel):
    run_id: str
    rollout_filename: str
    rollouts: list

class LoadRolloutRequest(BaseModel):
    rollout_filename: str

@app.post("/save_rollouts_data")
def save_rollouts_data(saveRolloutRequest: SaveRolloutRequest):
    run_id = saveRolloutRequest.run_id
    rollout_filename = sanitize_storage_name(
        saveRolloutRequest.rollout_filename or f"rollouts_{run_id}",
        ".json",
    )
    rollouts = saveRolloutRequest.rollouts
    with open(ROLLOUTS_DIR / rollout_filename, "w") as f:
        json.dump(rollouts, f)
    log_run_event(
        "rollouts_saved",
        run_id=run_id,
        details={
            "rollout_filename": rollout_filename,
            "rollout_count": len(rollouts),
            "path": str((ROLLOUTS_DIR / rollout_filename).resolve()),
        },
    )
    return {"status": "success", "run_id": run_id, "rollouts": rollouts}

@app.get("/rollouts_files")
def list_rollouts_files():
    files = [path.name for path in ROLLOUTS_DIR.iterdir() if path.is_file() and path.suffix == ".json"]
    return {"rollouts": sorted(files)}

@app.post("/load_rollouts_data")
def load_rollouts_data(loadRolloutRequest: LoadRolloutRequest):
    rollout_filename = sanitize_storage_name(loadRolloutRequest.rollout_filename, ".json")
    rollout_path = ROLLOUTS_DIR / rollout_filename
    if not rollout_path.exists():
        raise HTTPException(status_code=404, detail="Rollout file not found.")
    with open(rollout_path, "r") as f:
        rollouts = json.load(f)
    return {"status": "success", "rollout_filename": rollout_filename, "rollouts": rollouts}
class SessionState:
    def __init__(self):
        self.resume_event = asyncio.Event()
        self.resume_event.set()  # start un-paused
        self.paused = False
        self.model: Any = None

if HAS_MULTIPART:
    @app.post("/upload_model")
    async def upload_model(file: UploadFile = File(...)):
      if not file.filename or not file.filename.endswith(".zip"):
          return {"ok": False, "error": "must be a .zip"}
      safe_model_name = sanitize_storage_name(file.filename, ".zip")
      dest = MODELS_DIR / safe_model_name
      with open(dest, "wb") as f:
          f.write(await file.read())
      inferred_env_name = _infer_env_name_from_model_name(safe_model_name)
      if inferred_env_name:
          write_model_metadata(
              dest,
              env_name=inferred_env_name,
              run_id=None,
              source="upload_inferred",
          )
      return {"ok": True, "model_name": safe_model_name}
else:
    @app.post("/upload_model")
    async def upload_model():
      return {"ok": False, "error": "python-multipart is not installed in this backend environment"}

# sessions: dict[str, SessionState] = {}  # session_id -> state

import collections
current_model = collections.defaultdict(None)
current_model_locks = collections.defaultdict(Lock)
training_preview_model = collections.defaultdict(None)
training_preview_model_locks = collections.defaultdict(Lock)
class LoadRequest(BaseModel):
    run_id:str
    model_name:str
    env_name:str | None = None

class DeleteAllTempModelsRequest(BaseModel):
    run_id:str

@app.get("/get_model_path")
def get_model_path():
    return str(MODELS_DIR)

number_of_steps_dictionary = {}
class ChangeNumberOfStepsRequest(BaseModel):
    run_id: str
    number_of_steps: int
@app.post("/change_number_of_steps")
def change_number_of_steps(req: ChangeNumberOfStepsRequest):
    global number_of_steps_dictionary
    run_id = req.run_id
    number_of_steps = req.number_of_steps
    number_of_steps_dictionary[run_id] = number_of_steps
    return {"status": "success", "number_of_steps": number_of_steps}

@app.post("/load_model")
def load_model(req: LoadRequest):
    ensure_sb3_available()
    print("model loading began: ")
    if req.model_name == "":
        with current_model_locks[req.run_id]:
            current_model[req.run_id]  = None
        print("load model is None")
        return {"ok": True, "run_id": req.run_id, "model":""}
    path = MODELS_DIR / sanitize_storage_name(req.model_name, ".zip")
    if not path.exists():
        return {"ok": False, "error": "model_not_found"}
    requested_env_name = str(req.env_name or "").strip()
    metadata = read_model_metadata(path)
    model_env_name = str(metadata.get("env_name") or "").strip()
    if requested_env_name and model_env_name and requested_env_name != model_env_name:
        return {
            "ok": False,
            "error": "model_env_mismatch",
            "expected_env_name": requested_env_name,
            "model_env_name": model_env_name,
        }
    # Load the model
    # print("Sessions are this: ", sessions)
    # state = sessions.get(req.session_id)
    # print("State is this: ", state)
    # if state is None:
    #     return {"ok": False, "error": "session_not_found"}
    try:
        with current_model_locks[req.run_id]:
            current_model[req.run_id] = PPO.load(str(path))
        print("load model is", current_model)
        return {"ok": True, "run_id": req.run_id, "model": path.name, "model_env_name": model_env_name}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/delete_all_temp_models")
def delete_all_temp_models(req: DeleteAllTempModelsRequest):
    run_id = req.run_id
    for model_path in MODELS_DIR.iterdir():
        if model_path.is_file() and model_path.name.startswith(f"temp_{run_id}_") and model_path.suffix == ".zip":
            model_path.unlink()
    return {"ok": True, "run_id": run_id, "status": "all temp models deleted"}



@app.websocket("/ws/rollout")
async def rollout_stream(websocket: WebSocket):#, env_name:str = "CartPole-v1"):
    global env_name
    await websocket.accept()

    
    from ws_manager import WSManager 
    router = APIRouter()
    loop = asyncio.get_running_loop()
    ws_manager = WSManager(loop=loop)
    ws_manager.mark_loop_thread()

    # register some basic variables in relation to ws_manager and the websocket
    query = parse_qs(websocket.url.query)
    topic = "training" # only training for now
    print("query: ", query)
    run_id = query.get("runid", [None])[0]
    env_name = query.get("env", ["CartPole-v1"])[0]
    train_mode_str = query.get("train", ["true"])[0]
    train_mode = train_mode_str.lower() == "true"   # ✅ real boolean
    train_steps = query.get("train_steps", [1000])[0]
    train_steps = int(train_steps)
    ensure_run_status(run_id, env_name=env_name)

    
    client = await ws_manager.register(websocket, topic=topic, run_id=run_id)
    # Start the pump in the background (runs until cancelled/disconnect)
    # app.state.client = client
    # app.state.pump_task = asyncio.create_task(ws_manager.pump(client), name="pump")

    print("env_name: ", env_name)
    print("train_steps: ", train_steps)

    
    # generate + send a session ID
    session_id = generate_session_id()
    rollout_pause_state[session_id] = False  # default: not paused
    await websocket.send_json({"type": "session", "session_id": session_id})

    # get a query parameter
    #query = websocket.headers.get("sec-websocket-protocol", "CartPole-v1")
    #env_name = query or "CartPole-v1"
    
    def render_env(env):#mode="rgb_array"):
        try: 
            frame = env.render()
            _, buffer = cv2.imencode('.jpg', frame)
            #print(buffer.shape)
            return base64.b64encode(buffer).decode("utf-8")
        except Exception as e:
            print("Error rendering environment: ", e)
            return None

    try:
        
        env = make_reward_wrapped_env(env_name, run_id=run_id, render_mode="rgb_array")
        log_run_event(
            "rollout_session_started",
            run_id=run_id,
            env_name=env_name,
            details={
                "train_mode": bool(train_mode),
                "train_steps": int(train_steps),
                "send_frame_interval": int(number_of_steps_dictionary.get(run_id, 5)),
                "playback_delay_sec": float(time_intervals.get(run_id, 0.05)),
            },
        )
        log_reward_spec_snapshot(
            run_id=run_id,
            env_name=env_name,
            terms=get_reward_terms_for_run(run_id, env_name),
            task_config=get_task_config_for_run(run_id, env_name),
            source="rollout_session_start",
            details={"train_mode": bool(train_mode)},
        )
        #env = Monitor(env)
        model = None
        print("Model loaded: ", model)
        if train_mode:
            ensure_sb3_available()
            # Train
            callback = ProgressBarCallback(total_timesteps=train_steps, runId=run_id)
            global train_run_progress
            train_run_progress= 0

            model = start_training(run_id=run_id, env_name=env_name, train_steps=train_steps, reset_num_timesteps=False, callback=callback, 
                           ws_manager=ws_manager, frame_fn=render_env, every_n_steps=100)
        obs, _ = env.reset()
        step = 0
        episodes_seen = 0

        ep_reward = 0
        ep_reward_breakdown = {}
        ep_reward_raw_terms = {}
        ep_reward_history = []
        ep_behavior_trace = []
        send_frame_interval = 5 if run_id not in number_of_steps_dictionary else number_of_steps_dictionary[run_id]
        ep_frames = []

        # Script once outside the loop (TorchScript is optional but gives speedup)
        # if train_mode:
        #     with torch.no_grad():
        #         example_obs = torch.tensor(env.observation_space.sample(), dtype=torch.float32).unsqueeze(0).to(device)
        #         print("example observation shape: ", example_obs.shape)
        #         traced_policy = torch.jit.trace(model.policy, example_obs)
        while True:
            action = None
            while rollout_pause_state[session_id]:
                # supposed to keep looping until unpaused
                print("Rollout paused for 0.1 seconds")
                await asyncio.sleep(0.1)
            if train_mode:
                with torch.no_grad():
                    active_model = training_preview_model.get(run_id) or model
                    with training_preview_model_locks[run_id]:
                        action, _ = active_model.predict(obs, deterministic=True)
                    #print("example action output: ", action)
                    if isinstance(env.action_space, gym.spaces.Discrete):
                        action = int(np.asarray(action).reshape(-1)[0])
                    else:
                        action = action.squeeze(0)
                    
            else:
                if run_id not in current_model or current_model[run_id] is None:
                    action = env.action_space.sample()
                else:
                    #print("--- Using custom model right now ---")
                    with torch.no_grad():
                        with current_model_locks[run_id]:
                            action, _ = current_model[run_id].predict(obs, deterministic=True)
                        if isinstance(env.action_space, gym.spaces.Discrete):
                            action = int(np.asarray(action).reshape(-1)[0])
                        else:
                            action = action.squeeze(0)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_reward_breakdown = info.get("reward_breakdown", {"total": float(reward)})
            step_reward_raw_terms = info.get("reward_raw_terms", {"native": float(reward)})
            current_step = step + 1
            
            send_frame_interval = 5 if run_id not in number_of_steps_dictionary else number_of_steps_dictionary[run_id]
            capture_episode_frames = episodes_seen % send_frame_interval == 0
            ep_reward += reward
            for key, value in step_reward_breakdown.items():
                ep_reward_breakdown[key] = ep_reward_breakdown.get(key, 0.0) + float(value)
            for key, value in step_reward_raw_terms.items():
                ep_reward_raw_terms[key] = ep_reward_raw_terms.get(key, 0.0) + float(value)
            ep_reward_history.append({
                "step": current_step,
                "reward": float(reward),
                "reward_breakdown": {key: float(value) for key, value in step_reward_breakdown.items()},
                "reward_raw_terms": {key: float(value) for key, value in step_reward_raw_terms.items()},
            })
            ep_behavior_trace.append({
                "step": current_step,
                "observation": np.asarray(next_obs, dtype=np.float32).tolist(),
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "time_sec": float(current_step),
            })
            if capture_episode_frames:
                ep_frames.append(render_env(env))

            # Prepare for next step
            if done:
                frames = ep_frames if episodes_seen % send_frame_interval == 0 else []
                sim_frame_episode_number = episodes_seen if episodes_seen % send_frame_interval == 0 else None
                # Construct the data payload
                data = {
                    #"step": step,
                    "type": "rollout",
                    "episode": episodes_seen,
                    #"observation": obs.tolist(),
                    #"action": int(action),
                    "sim_frame_episode_number": sim_frame_episode_number,
                    "ep_frames": frames,
                    "reward": float(ep_reward),
                    "reward_breakdown": {key: float(value) for key, value in ep_reward_breakdown.items()},
                    "reward_raw_terms": {key: float(value) for key, value in ep_reward_raw_terms.items()},
                    "reward_history": ep_reward_history,
                    "episode_outcome": info.get("episode_outcome", "unknown"),
                    "episode_outcome_reason": info.get("episode_outcome_reason", "outcome unavailable"),
                    "episode_terminal_timestep": int(info.get("episode_terminal_timestep", len(ep_reward_history))),
                    "terminated": bool(info.get("episode_terminated", terminated)),
                    "truncated": bool(info.get("episode_truncated", truncated)),
                    "reward_weights": reward_weights_for_run(run_id, env_name),
                    #"done": done
                }
                backend_episode_payload = {**data, "behavior_trace": list(ep_behavior_trace)}
                append_recent_episode(run_id, backend_episode_payload, source="rollout", env_name=env_name)
                #print("Rollout data sent with reward: ", float(ep_reward));

                # Send JSON over WebSocket
                await websocket.send_text(json.dumps(data))
                #print("Sent data: ", data)
                obs, _ = env.reset()
                step = 0
                ep_reward = 0
                ep_reward_breakdown = {}
                ep_reward_raw_terms = {}
                ep_reward_history = []
                ep_behavior_trace = []
                episodes_seen += 1
                ep_frames = []
            else:
                obs = next_obs
                step += 1

            await asyncio.sleep(time_intervals[run_id] if (run_id in time_intervals) else 0.05)  # throttle to ~20 FPS
    except WebSocketDisconnect:
        print("Client disconnected. ")
    finally:
        if "env" in locals():
            env.close()


@app.api_route("/healthz", methods=["GET", "HEAD"])
def healthcheck():
    return {
        "ok": True,
        "status": "healthy",
        "timestamp": time.time(),
    }


def _frontend_file_response(relative_path: str) -> FileResponse:
    target_path = (FRONTEND_DIST_DIR / relative_path).resolve()
    try:
        target_path.relative_to(FRONTEND_DIST_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="frontend asset not found") from exc
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="frontend asset not found")
    return FileResponse(target_path)


@app.get("/")
def serve_frontend_index():
    if not FRONTEND_DIST_DIR.exists():
        return {
            "message": "Frontend build not found. Run `npm install && npm run build` inside `opengym-frontend` before starting production.",
            "api": "OpenGym Copilot backend is running.",
        }
    return FileResponse(FRONTEND_DIST_DIR / "index.html")


@app.get("/{full_path:path}")
def serve_frontend_assets(full_path: str):
    if not FRONTEND_DIST_DIR.exists():
        raise HTTPException(status_code=404, detail="frontend build not found")
    if full_path.startswith(("models", "rollouts", "progress", "reward_config", "training_runs", "ws")):
        raise HTTPException(status_code=404, detail="not found")
    target = (FRONTEND_DIST_DIR / full_path).resolve()
    if target.exists() and target.is_file():
        return _frontend_file_response(full_path)
    return FileResponse(FRONTEND_DIST_DIR / "index.html")
