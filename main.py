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

device = "cuda" if torch.cuda.is_available() else "cpu"
time_intervals = {}
rollout_fps = {}

BASE_DIR = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env", override=False)
DATA_DIR = Path(os.getenv("OPEN_GYM_DATA_DIR", str(BASE_DIR))).resolve()
MODELS_DIR = (DATA_DIR / "models").resolve()
ROLLOUTS_DIR = (DATA_DIR / "rollouts").resolve()
PROGRESS_LOG_PATH = (DATA_DIR / "progress_bar.log").resolve()
FRONTEND_DIST_DIR = (BASE_DIR / "opengym-frontend" / "dist").resolve()

for path in (DATA_DIR, MODELS_DIR, ROLLOUTS_DIR):
    path.mkdir(parents=True, exist_ok=True)

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
    safe_env_name = (selected_env_name or "env").replace("/", "_")
    return str((MODELS_DIR / f"ppo_model_{safe_env_name}_{timestamp}.zip").resolve())


def sanitize_storage_name(name: str | None, suffix: str) -> str:
    raw_name = (name or "").strip()
    candidate = Path(raw_name).name
    if not candidate:
        candidate = f"default{suffix}"
    if not candidate.endswith(suffix):
        candidate = f"{candidate}{suffix}"
    return candidate


def resolve_model_output_path(requested_path: str | None, run_id: str | None, selected_env_name: str) -> Path:
    default_name = sanitize_storage_name(f"ppo_model_{selected_env_name}_{run_id or 'session'}", ".zip")
    if not requested_path or not requested_path.strip():
        return (MODELS_DIR / default_name).resolve()

    candidate = Path(requested_path.strip())
    if candidate.is_absolute():
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = (MODELS_DIR / default_name).resolve()
        else:
            try:
                resolved.relative_to(MODELS_DIR)
            except ValueError:
                resolved = (MODELS_DIR / sanitize_storage_name(candidate.name, ".zip")).resolve()
    else:
        safe_parts = [part for part in candidate.parts if part not in {"", ".", ".."}]
        resolved = (MODELS_DIR / Path(*safe_parts)).resolve() if safe_parts else (MODELS_DIR / default_name).resolve()

    if resolved.suffix.lower() != ".zip":
        resolved = resolved.with_suffix(".zip")
    try:
        resolved.relative_to(MODELS_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="training output path must stay inside the managed models directory") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved

def encode_jpeg(frame_np, quality=80):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame_np).save(buf, format="jpeg", quality=quality)
    return buf.getvalue()


class RewardTermPayload(BaseModel):
    key: str
    weight: float
    enabled: bool = True
    label: str | None = None
    description: str | None = None
    expression: str | None = None


class RewardConfigUpdateRequest(BaseModel):
    run_id: str
    env_name: str
    terms: list[RewardTermPayload]


class TaskConfigProposalRequest(BaseModel):
    run_id: str
    env_name: str
    goal: str


class TaskConfigApplyRequest(BaseModel):
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


def update_task_config_request_status(run_id: str | None, **fields: Any) -> None:
    if not run_id:
        return
    current = task_config_request_status_by_run.get(run_id, {})
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
    cached = reward_variable_cache.get(env_name)
    if cached is not None:
        return cached

    env = gym.make(env_name)
    try:
        obs_space = env.observation_space
        action_space = env.action_space

        obs_size = int(np.prod(obs_space.shape)) if getattr(obs_space, "shape", None) else 1
        if getattr(action_space, "shape", None):
            action_size = int(np.prod(action_space.shape))
        else:
            action_size = 1

        raw_term_keys = [term["key"] for term in reward_template_for_env(env_name)]
        specs = reward_expression_variable_specs(
            env_name=env_name,
            obs_size=obs_size,
            action_size=action_size,
            raw_term_keys=raw_term_keys,
        )
        reward_variable_cache[env_name] = specs
        return specs
    finally:
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
    }
    task_configs_by_run[run_id] = normalized
    return normalized


def validate_task_config_for_run(config: dict[str, Any], env_name: str, run_id: str | None = None) -> dict[str, Any]:
    base_names = reward_variable_names_for_env(env_name)
    param_names: list[str] = []
    seen_param_names: set[str] = set()
    for param in config.get("task_params", []) or []:
        key = str(param.get("key") or "").strip()
        if not key:
            continue
        if key in seen_param_names:
            raise ValueError(f"Duplicate task parameter '{key}'.")
        seen_param_names.add(key)
        param_names.append(key)

    derived_names: list[str] = []
    seen_signal_names: set[str] = set()
    zero_context = {name: 0.0 for name in [*base_names, *param_names]}
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
    return variable_names


def _extract_first_number(text: str, default: float) -> float:
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return float(default)
    try:
        return float(match.group(1))
    except ValueError:
        return float(default)


def build_heuristic_task_config_proposal(goal: str, env_name: str) -> dict[str, Any]:
    normalized_goal = str(goal or "").strip()
    lower_goal = normalized_goal.lower()
    terms = reward_template_for_env(env_name)
    task_params: list[dict[str, Any]] = []
    derived_signals: list[dict[str, Any]] = []
    warnings: list[str] = []
    rationale = "Heuristic fallback proposal based on the current environment schema and goal keywords."
    success_metric = "Long episodes with stable shaped reward and fewer late-episode failure spikes."

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
        "task_params": task_params,
        "derived_signals": derived_signals,
        "reward_terms": terms,
        "success_metric": success_metric,
        "rationale": rationale,
        "warnings": warnings,
        "provider": "heuristic",
        "model": "",
    }


def call_llm_task_config_proposal(goal: str, env_name: str, *, run_id: str | None = None) -> dict[str, Any]:
    api_key = (
        os.getenv("TASK_CONFIG_LLM_API_KEY", "").strip()
        or os.getenv("ZAI_API_KEY", "").strip()
        or os.getenv("BIGMODEL_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("No task-config LLM API key is configured. Set TASK_CONFIG_LLM_API_KEY or ZAI_API_KEY.")

    base_url = os.getenv("TASK_CONFIG_LLM_BASE_URL", "https://api.z.ai/api/paas/v4").strip().rstrip("/")
    model = os.getenv("TASK_CONFIG_LLM_MODEL", "glm-5").strip() or "glm-5"
    timeout_sec = max(15, int(float(os.getenv("TASK_CONFIG_LLM_TIMEOUT_SEC", "120"))))
    max_retries = max(1, int(os.getenv("TASK_CONFIG_LLM_MAX_RETRIES", "2")))
    variable_specs = task_variable_specs_for_run(None, env_name)
    current_terms = reward_template_for_env(env_name)
    started_at = time.time()
    update_task_config_request_status(
        run_id,
        status="preparing",
        message=f"Preparing LLM task-config request for model={model}.",
        model=model,
        base_url=base_url,
        env_name=env_name,
        goal=goal,
        attempt=0,
        elapsed_sec=0.0,
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["goal", "task_params", "derived_signals", "reward_terms", "success_metric", "rationale", "warnings"],
        "properties": {
            "goal": {"type": "string"},
            "task_params": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["key", "value", "description"],
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": "number"},
                        "description": {"type": "string"},
                    },
                },
            },
            "derived_signals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["key", "expression", "description"],
                    "properties": {
                        "key": {"type": "string"},
                        "expression": {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
            },
            "reward_terms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["key", "label", "description", "weight", "enabled", "expression"],
                    "properties": {
                        "key": {"type": "string"},
                        "label": {"type": "string"},
                        "description": {"type": "string"},
                        "weight": {"type": "number"},
                        "enabled": {"type": "boolean"},
                        "expression": {"type": "string"},
                    },
                },
            },
            "success_metric": {"type": "string"},
            "rationale": {"type": "string"},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
    }
    system_prompt = (
        "You are proposing a structured RL task config. Return valid JSON only. "
        "Use only the provided formula variables and helper math functions already supported by the backend: "
        "abs, min, max, clip, sqrt, square, exp, log, sin, cos, tanh, sign. "
        "Reward term expressions must be runnable immediately; if you suggest task parameters or derived signals, "
        "reward terms should still use variables that are already supported now, such as time_sec or step."
    )
    user_payload = {
        "env_name": env_name,
        "goal": goal,
        "available_variables": variable_specs,
        "default_reward_terms": current_terms,
        "formula_examples": reward_formula_examples_for_env(env_name),
        "output_requirements": {
            "must_return_json_object": True,
            "top_level_keys": ["goal", "task_params", "derived_signals", "reward_terms", "success_metric", "rationale", "warnings"],
            "reward_term_fields": ["key", "label", "description", "weight", "enabled", "expression"],
        },
    }
    prompt = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(user_payload),
            },
        ],
        "response_format": {"type": "json_object"},
    }
    request = urllib_request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(prompt).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        update_task_config_request_status(
            run_id,
            status="sending",
            message=f"Sending task-config request to {base_url} with model={model} (attempt {attempt}/{max_retries}).",
            attempt=attempt,
            elapsed_sec=round(time.time() - started_at, 2),
        )
        try:
            with urllib_request.urlopen(request, timeout=timeout_sec) as response:
                update_task_config_request_status(
                    run_id,
                    status="waiting_response_body",
                    message=f"Connected to provider. Reading response body for model={model}.",
                    attempt=attempt,
                    elapsed_sec=round(time.time() - started_at, 2),
                )
                body = json.loads(response.read().decode("utf-8"))
            update_task_config_request_status(
                run_id,
                status="received",
                message=f"Received response from provider for model={model}. Parsing proposal.",
                attempt=attempt,
                elapsed_sec=round(time.time() - started_at, 2),
            )
            break
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            update_task_config_request_status(
                run_id,
                status="http_error",
                message=f"Provider returned HTTP error for model={model}: {detail[:300]}",
                attempt=attempt,
                elapsed_sec=round(time.time() - started_at, 2),
            )
            raise RuntimeError(
                f"LLM task proposal request failed for model={model} base_url={base_url}: {detail}"
            ) from exc
        except (urllib_error.URLError, TimeoutError, socket.timeout) as exc:
            last_error = exc
            if attempt >= max_retries:
                reason = getattr(exc, "reason", None) or str(exc)
                update_task_config_request_status(
                    run_id,
                    status="timeout",
                    message=(
                        f"Request timed out/failed after {attempt} attempt(s) for model={model} "
                        f"at {base_url} with timeout={timeout_sec}s."
                    ),
                    attempt=attempt,
                    elapsed_sec=round(time.time() - started_at, 2),
                )
                raise RuntimeError(
                    f"LLM task proposal request timed out/failed after {attempt} attempt(s) "
                    f"for model={model} base_url={base_url} timeout={timeout_sec}s: {reason}"
                ) from exc
            update_task_config_request_status(
                run_id,
                status="retrying",
                message=(
                    f"Attempt {attempt}/{max_retries} failed for model={model}. "
                    f"Retrying after backoff."
                ),
                attempt=attempt,
                elapsed_sec=round(time.time() - started_at, 2),
            )
            time.sleep(min(2 * attempt, 5))
    else:
        raise RuntimeError(
            f"LLM task proposal request failed for model={model} base_url={base_url}: {last_error}"
        )

    output_text = None
    choices = body.get("choices") or []
    if choices:
        output_text = choices[0].get("message", {}).get("content")
    if not output_text:
        update_task_config_request_status(
            run_id,
            status="parse_error",
            message=f"Provider response did not include text content for model={model}.",
            attempt=max_retries,
            elapsed_sec=round(time.time() - started_at, 2),
        )
        raise RuntimeError("LLM response did not include structured output text.")
    proposal = json.loads(output_text)
    provider_label = "glm"
    if "z.ai" in base_url or "bigmodel" in base_url:
        provider_label = "glm"
    elif "openai" in base_url:
        provider_label = "openai-compatible"
    else:
        provider_label = "openai-compatible"
    proposal["provider"] = provider_label
    proposal["model"] = model
    update_task_config_request_status(
        run_id,
        status="completed",
        message=f"Task-config proposal completed successfully with model={model}.",
        attempt=max_retries,
        elapsed_sec=round(time.time() - started_at, 2),
    )
    return proposal


def propose_task_config(goal: str, env_name: str, *, run_id: str | None = None) -> dict[str, Any]:
    try:
        proposal = call_llm_task_config_proposal(goal, env_name, run_id=run_id)
    except Exception as exc:
        proposal = build_heuristic_task_config_proposal(goal, env_name)
        proposal.setdefault("warnings", [])
        proposal["warnings"] = [f"Model proposal unavailable; used heuristic fallback instead. {exc}", *proposal["warnings"]]
        update_task_config_request_status(
            run_id,
            status="fallback",
            message=str(proposal["warnings"][0]),
            elapsed_sec=task_config_request_status_by_run.get(run_id, {}).get("elapsed_sec", 0.0),
        )
    try:
        normalized_terms = validate_reward_terms(
            env_name,
            proposal.get("reward_terms") or [],
            available_variable_names=task_variable_names_for_run(None, env_name),
        )
        proposal["reward_terms"] = normalized_terms
    except Exception as exc:
        fallback = build_heuristic_task_config_proposal(goal, env_name)
        fallback.setdefault("warnings", [])
        fallback["warnings"] = [f"Non-runnable proposal discarded; used heuristic fallback instead. {exc}", *fallback["warnings"]]
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
    if env_name:
        status["env_name"] = env_name
    return status


def append_recent_episode(run_id: str | None, episode_payload: dict[str, Any], source: str, env_name: str | None = None) -> None:
    status = ensure_run_status(run_id, env_name=env_name)
    if not status:
        return
    key = "recent_training_episodes" if source == "training" else "recent_rollout_episodes"
    insight_key = "training_insights" if source == "training" else "rollout_insights"
    episodes = [episode_payload, *status.get(key, [])][:25]
    status[key] = episodes
    status[insight_key] = build_episode_insights(
        episodes,
        source=source,
        env_name=status.get("env_name"),
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
    return normalized


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
            RUNS_TRAINING_STATUS[run_id]["model_path"] = train_model_actual_path
            RUNS_TRAINING_STATUS[run_id]["status"] = "stopped" if stop_requested else "done"
            print("Training stopped early" if stop_requested else "Training complete")
        except Exception as e:
            RUNS_TRAINING_STATUS[run_id]["status"] = "error"
            RUNS_TRAINING_STATUS[run_id]["error"] = str(e)
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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    files = [path.name for path in MODELS_DIR.iterdir() if path.is_file() and path.suffix == ".zip"]
    return {"models": sorted(files)}

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
    if status.get("recent_training_episodes") and not status.get("training_insights"):
        status["training_insights"] = build_episode_insights(
            status.get("recent_training_episodes", []),
            source="training",
            env_name=status.get("env_name"),
        )
    if status.get("recent_rollout_episodes") and not status.get("rollout_insights"):
        status["rollout_insights"] = build_episode_insights(
            status.get("recent_rollout_episodes", []),
            source="rollout",
            env_name=status.get("env_name"),
        )
    return status


@app.get("/reward_config")
def get_reward_config(run_id: str, env_name: str):
    terms = get_reward_terms_for_run(run_id, env_name)
    task_config = get_task_config_for_run(run_id, env_name)
    return {
        "run_id": run_id,
        "env_name": env_name,
        "terms": terms,
        "supports_custom_reward": len(terms) > 1,
        "available_variables": task_variable_specs_for_run(run_id, env_name),
        "formula_examples": reward_formula_examples_for_env(env_name),
        "task_config": task_config,
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
    try:
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
    )
    proposal = propose_task_config(req.goal, req.env_name, run_id=req.run_id)
    return {
        "run_id": req.run_id,
        "env_name": req.env_name,
        **proposal,
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
        return {"ok": True, "run_id": req.run_id, "model": path.name}
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
                append_recent_episode(run_id, data, source="rollout", env_name=env_name)
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


@app.get("/healthz")
def healthcheck():
    return {
        "ok": True,
        "models_dir": str(MODELS_DIR),
        "rollouts_dir": str(ROLLOUTS_DIR),
        "frontend_built": FRONTEND_DIST_DIR.exists(),
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
