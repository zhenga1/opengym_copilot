from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import gymnasium as gym
from pydantic import BaseModel
from pyparsing import Optional
import torch
import json
import numpy as np
import cv2
import base64
from typing import Any
import importlib.util
#import constants

from urllib.parse import parse_qs

from train_backend_reward_tuning.reward_shaping import (
    RewardShapingWrapper,
    normalize_reward_terms,
    reward_monitor_keys,
    reward_template_for_env,
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
RUNS_TRAINING_STATUS = {}  # run_id -> {"status": "running|done|error", "model_path": str|None, ...}
reward_configs_by_run = {}  # run_id -> {"env_name": str, "terms": list[dict]}


app = FastAPI()
env_name = "" # unknown for now

device = "cuda" if torch.cuda.is_available() else "cpu"
time_intervals = {}
rollout_fps = {}

progress_bar_log_file = open("progress_bar.log", "w")
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
default_model_path = f"models/ppo_model_{env_name}_{timestamp}.zip"

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


class RewardConfigUpdateRequest(BaseModel):
    run_id: str
    env_name: str
    terms: list[RewardTermPayload]


def get_reward_terms_for_run(run_id: str | None, env_name: str) -> list[dict]:
    if not run_id:
        return reward_template_for_env(env_name)

    current = reward_configs_by_run.get(run_id)
    if current is None or current.get("env_name") != env_name:
        reward_configs_by_run[run_id] = {
            "env_name": env_name,
            "terms": reward_template_for_env(env_name),
        }

    stored_terms = reward_configs_by_run[run_id]["terms"]
    normalized = normalize_reward_terms(env_name, stored_terms)
    reward_configs_by_run[run_id]["terms"] = normalized
    return normalized


def set_reward_terms_for_run(run_id: str, env_name: str, terms: list[dict]) -> list[dict]:
    normalized = normalize_reward_terms(env_name, terms)
    reward_configs_by_run[run_id] = {"env_name": env_name, "terms": normalized}
    return normalized


def reward_weights_for_run(run_id: str | None, env_name: str) -> dict[str, float]:
    return {
        term["key"]: float(term["weight"]) if term["enabled"] else 0.0
        for term in get_reward_terms_for_run(run_id, env_name)
    }


def make_reward_wrapped_env(env_name: str, run_id: str | None, render_mode: str | None = None):
    env_kwargs = {"render_mode": render_mode} if render_mode else {}
    env = gym.make(env_name, **env_kwargs)
    return RewardShapingWrapper(
        env,
        env_name=env_name,
        config_provider=lambda: get_reward_terms_for_run(run_id, env_name),
    )


def make_training_env_factory(env_name: str, run_id: str | None):
    ensure_sb3_available()
    info_keywords = reward_monitor_keys(env_name)

    def _factory():
        env = make_reward_wrapped_env(env_name, run_id=run_id, render_mode=None)
        return Monitor(env, info_keywords=info_keywords)

    return _factory


#@app.post("/start")
def start_training(run_id: str = None, env_name: str = "CartPole-v1", train_steps:int = 1000, reset_num_timesteps = False, callback: Any = None,
                   ws_manager=None, frame_fn=None, every_n_steps:int = 100, model: Any = None):
    ensure_sb3_available()
    RUNS_TRAINING_STATUS.setdefault(run_id, {"status": "running", "model_path": None, "error": None})
    preview_model = None
    eval_env = None

    if model is None:
        vec_env = DummyVecEnv(
            [make_training_env_factory(env_name, run_id=run_id) for _ in range(8)]
        )

        device = "cpu"
        if run_id in training_model_devices:
            device = training_model_devices[run_id]

        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device=device,
            tensorboard_log="./tensorboard_logs"
        )

        preview_env = DummyVecEnv(
            [make_training_env_factory(env_name, run_id=run_id)]
        )
        preview_model = PPO(
            "MlpPolicy",
            preview_env,
            verbose=0,
            device="cpu",
        )
        preview_model.policy.load_state_dict(model.policy.state_dict())
        current_model[run_id] = preview_model
        eval_env = make_reward_wrapped_env(env_name, run_id=run_id, render_mode=None)
    else:
        current_model[run_id] = model

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
            with current_model_locks[self.run_id]:
                self.target_model.policy.load_state_dict(self.source_model.policy.state_dict())

        def _evaluate_once(self):
            if self.target_model is None or self.eval_env is None or self.status_dict is None:
                return
            obs, _ = self.eval_env.reset()
            total_reward = 0.0
            for _ in range(1000):
                with current_model_locks[self.run_id]:
                    action, _ = self.target_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            self.status_dict[self.run_id]["eval_reward"] = float(total_reward)

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
            model.learn(total_timesteps=train_steps, reset_num_timesteps=reset_num_timesteps, callback=callbacks_list)
            train_model_actual_path = ""
            if run_id not in trained_model_paths:
                train_model_actual_path = default_model_path
            else:
                train_model_actual_path = trained_model_paths[run_id]
            model.save(train_model_actual_path)
            RUNS_TRAINING_STATUS[run_id]["status"] = "done"
            RUNS_TRAINING_STATUS[run_id]["model_path"] = train_model_actual_path
            print("Training complete")
        except Exception as e:
            RUNS_TRAINING_STATUS[run_id]["status"] = "error"
            RUNS_TRAINING_STATUS[run_id]["error"] = str(e)
        finally:
            model_env = model.get_env()
            if model_env is not None:
                model_env.close()
            if eval_env is not None:
                eval_env.close()
    Thread(target=train).start()
    # status option
    return model

import os
from fastapi import UploadFile, File
MODELS_DIR = os.path.join(os.getcwd(), "models")
print("Models directory: ", MODELS_DIR)
os.makedirs(MODELS_DIR, exist_ok=True)
@app.get("/models")
def list_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.zip')]
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
    return RUNS_TRAINING_STATUS.get(run_id, {"status": "unknown"})


@app.get("/reward_config")
def get_reward_config(run_id: str, env_name: str):
    terms = get_reward_terms_for_run(run_id, env_name)
    return {
        "run_id": run_id,
        "env_name": env_name,
        "terms": terms,
        "supports_custom_reward": len(terms) > 1,
    }


@app.post("/reward_config")
def update_reward_config(req: RewardConfigUpdateRequest):
    terms = set_reward_terms_for_run(
        run_id=req.run_id,
        env_name=req.env_name,
        terms=[term.model_dump() for term in req.terms],
    )
    return {
        "status": "updated",
        "run_id": req.run_id,
        "env_name": req.env_name,
        "terms": terms,
    }



class PauseRequest(BaseModel):
    paused: bool
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

@app.post("/set_training_dir")
def set_training_dir(req: SetTrainingDirRequest):
    run_id = req.run_id
    where_to_save_trained_model = req.train_dir_path

    print("New parameter obtained: Here is where save trained model - ", where_to_save_trained_model)
    device = req.device
    print("Device obtainied ", device)
    trained_model_paths[run_id] = where_to_save_trained_model
    training_model_devices[run_id] = device
    # Here you would typically set the training directory for the session
    return {"status": "training directory set", "run_id": run_id, "path": where_to_save_trained_model}

class SaveRolloutRequest(BaseModel):
    run_id: str
    rollout_filename: str
    rollouts: list

class LoadRolloutRequest(BaseModel):
    rollout_filename: str

rollouts_dir = os.path.join(os.getcwd(), "rollouts")
os.makedirs(rollouts_dir, exist_ok=True)
@app.post("/save_rollouts_data")
def save_rollouts_data(saveRolloutRequest: SaveRolloutRequest):
    run_id = saveRolloutRequest.run_id
    rollout_filename = saveRolloutRequest.rollout_filename
    if rollout_filename == "" or rollout_filename is None or (len(rollout_filename)>=4 and ".json" not in rollout_filename):
        #print(f"Rollout filename: {rollout_filename} is invalid, please check. ")
        rollout_filename = f"rollouts_{run_id}.json"
    rollouts = saveRolloutRequest.rollouts
    with open(os.path.join(rollouts_dir, rollout_filename), "w") as f:
        json.dump(rollouts, f)
    return {"status": "success", "run_id": run_id, "rollouts": rollouts}

@app.get("/rollouts_files")
def list_rollouts_files():
    files = [f for f in os.listdir(rollouts_dir) if f.endswith(".json")]
    return {"rollouts": sorted(files)}

@app.post("/load_rollouts_data")
def load_rollouts_data(loadRolloutRequest: LoadRolloutRequest):
    rollout_filename = os.path.basename(loadRolloutRequest.rollout_filename)
    rollout_path = os.path.join(rollouts_dir, rollout_filename)
    if not os.path.exists(rollout_path):
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
      if not file.filename.endswith(".zip"):
          return {"ok": False, "error": "must be a .zip"}
      dest = os.path.join(MODELS_DIR, file.filename)
      with open(dest, "wb") as f:
          f.write(await file.read())
      return {"ok": True, "model_name": file.filename}
else:
    @app.post("/upload_model")
    async def upload_model():
      return {"ok": False, "error": "python-multipart is not installed in this backend environment"}

# sessions: dict[str, SessionState] = {}  # session_id -> state

import collections
current_model = collections.defaultdict(None)
current_model_locks = collections.defaultdict(Lock)
class LoadRequest(BaseModel):
    run_id:str
    model_name:str

class DeleteAllTempModelsRequest(BaseModel):
    run_id:str

@app.get("/get_model_path")
def get_model_path():
    return MODELS_DIR

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
    from os.path import join, exists
    print("model loading began: ")
    if req.model_name == "":
        with current_model_locks[req.run_id]:
            current_model[req.run_id]  = None
        print("load model is None")
        return {"ok": True, "run_id": req.run_id, "model":""}
    path = join(MODELS_DIR, req.model_name)
    if not exists(path):
        return {"ok": False, "error": "model_not_found"}
    # Load the model
    # print("Sessions are this: ", sessions)
    # state = sessions.get(req.session_id)
    # print("State is this: ", state)
    # if state is None:
    #     return {"ok": False, "error": "session_not_found"}
    try:
        with current_model_locks[req.run_id]:
            current_model[req.run_id] = PPO.load(path)
        print("load model is", current_model)
        return {"ok": True, "run_id": req.run_id, "model": req.model_name}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/delete_all_temp_models")
def delete_all_temp_models(req: DeleteAllTempModelsRequest):
    run_id = req.run_id
    for filename in os.listdir(MODELS_DIR):
        if filename.startswith(f"temp_{run_id}_") and filename.endswith(".zip"):
            os.remove(os.path.join(MODELS_DIR, filename))
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
        frame = env.render()
        _, buffer = cv2.imencode('.jpg', frame)
        #print(buffer.shape)
        return base64.b64encode(buffer).decode("utf-8")

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
                    active_model = current_model.get(run_id) or model
                    with current_model_locks[run_id]:
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
            
            send_frame_interval = 5 if run_id not in number_of_steps_dictionary else number_of_steps_dictionary[run_id]
            ep_reward += reward
            for key, value in step_reward_breakdown.items():
                ep_reward_breakdown[key] = ep_reward_breakdown.get(key, 0.0) + float(value)
            for key, value in step_reward_raw_terms.items():
                ep_reward_raw_terms[key] = ep_reward_raw_terms.get(key, 0.0) + float(value)

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
                    "reward_weights": reward_weights_for_run(run_id, env_name),
                    #"done": done
                }
                #print("Rollout data sent with reward: ", float(ep_reward));

                # Send JSON over WebSocket
                await websocket.send_text(json.dumps(data))
                #print("Sent data: ", data)
                obs, _ = env.reset()
                step = 0
                ep_reward = 0
                ep_reward_breakdown = {}
                ep_reward_raw_terms = {}
                episodes_seen += 1
                ep_frames = []
            else:
                if episodes_seen % send_frame_interval == 0 :
                    ep_frames.append(render_env(env))
                obs = next_obs
                step += 1

            await asyncio.sleep(time_intervals[run_id] if (run_id in time_intervals) else 0.05)  # throttle to ~20 FPS
    except WebSocketDisconnect:
        print("Client disconnected. ")
