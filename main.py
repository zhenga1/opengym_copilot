from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import gymnasium as gym
from pyparsing import Optional
import torch
from stable_baselines3 import PPO
import json
import numpy as np
import cv2
import base64
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
#import constants

from urllib.parse import parse_qs

from train_backend_reward_tuning.cartpole_dance_left_right import CartPoleDanceWrapper


trained_model_paths = {} # run_id to most recent saved model paths
training_model_devices = {} # run_id to device to use
RUNS_TRAINING_STATUS = {}  # run_id -> {"status": "running|done|error", "model_path": str|None, ...}


app = FastAPI()
env_name = "" # unknown for now

device = "cuda" if torch.cuda.is_available() else "cpu"

progress_bar_log_file = open("progress_bar.log", "w")
train_run_progresses = {}
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, runId, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.runId = runId
    
    def _on_step(self) -> bool:
        global train_run_progress
        # bound the progress bar percentage by the minimum of n_calls and total_timesteps, so self.n_calls never exceeds self.total_timesteps
        pct = 100 * min(self.n_calls, self.total_timesteps) / self.total_timesteps
        # file.write(f"Progress: {pct:.2f}%\n")
        # file.write(f"n_calls: {self.n_calls}\n total_timesteps: {self.total_timesteps}\n")
        # file.flush()
        print(f"Progress: {pct:.2f}%", end='\r') # or send to the frontend
        
        train_run_progresses[self.runId] = pct
        return True
    
from fastapi import WebSocket
from threading import Thread
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
default_model_path = f"models/ppo_model_{env_name}_{timestamp}.zip"
#@app.post("/start")
def start_training(model: PPO = None, run_id: str = None, train_steps:int = 1000, reset_num_timesteps = False,callback: BaseCallback = None):
    RUNS_TRAINING_STATUS.setdefault(run_id, {"status": "running", "model_path": None, "error": None})
    def train():
        global train_run_progress, RUNS_TRAINING_STATUS
        train_run_progress = 0
        try:
            model.learn(total_timesteps=train_steps, reset_num_timesteps=False, callback=callback)
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
    Thread(target=train).start()
    # status option
    return {"status": "training started"}

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


@app.get("/training_runs/{run_id}")
def get_run(run_id: str):
    return RUNS_TRAINING_STATUS.get(run_id, {"status": "unknown"})

from pydantic import BaseModel
class PauseRequest(BaseModel):
    paused: bool
@app.get("/progress/{run_id}")
def get_progress(run_id: str):
    #return {"progress": train_run_progresses.get(run_id, 0)}
    #     await asyncio.sleep(0.1)
    # await websocket.close()
    progress_bar_log_file.write(f"Put variable by name progress {train_run_progresses} for run_id {run_id}\n")
    return {"progress": train_run_progresses.get(run_id, 0)}

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


class SessionState:
    def __init__(self):
        self.resume_event = asyncio.Event()
        self.resume_event.set()  # start un-paused
        self.paused = False
        self.model:Optional[PPO] = None

@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
  if not file.filename.endswith(".zip"):
      return {"ok": False, "error": "must be a .zip"}
  dest = os.path.join(MODELS_DIR, file.filename)
  with open(dest, "wb") as f:
      f.write(await file.read())
  return {"ok": True, "model_name": file.filename}

# sessions: dict[str, SessionState] = {}  # session_id -> state

import collections
current_model = collections.defaultdict(None)
class LoadRequest(BaseModel):
    run_id:str
    model_name:str

@app.get("/get_model_path")
def get_model_path():
    return MODELS_DIR
@app.post("/load_model")
def load_model(req: LoadRequest):
    from os.path import join, exists
    print("model loading began: ")
    if req.model_name == "":
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
        current_model[req.run_id] = PPO.load(path)
        print("load model is", current_model)
        return {"ok": True, "run_id": req.run_id, "model": req.model_name}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    

@app.websocket("/ws/training")
async def training_procedure(websocket: WebSocket):
    await websocket.accept()
    while True:
        message = await websocket.receive_text()
        if message == "pause":
            # Pause the training
            await websocket.send_text("Training paused")
        elif message == "resume":
            # Resume the training
            await websocket.send_text("Training resumed")
        elif message == "stop":
            # Stop the training
            await websocket.send_text("Training stopped")
            break

@app.websocket("/ws/rollout")
async def rollout_stream(websocket: WebSocket):#, env_name:str = "CartPole-v1"):
    global env_name
    await websocket.accept()


    query = parse_qs(websocket.url.query)
    print("query: ", query)
    run_id = query.get("runid", [None])[0]
    env_name = query.get("env", ["CartPole-v1"])[0]
    train_mode_str = query.get("train", ["true"])[0]
    train_mode = train_mode_str.lower() == "true"   # ✅ real boolean
    train_steps = query.get("train_steps", [1000])[0]
    train_steps = int(train_steps)

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
        
        env = gym.make(env_name, render_mode="rgb_array")
        model = None
        print("Model loaded: ", model)
        if train_mode:
            # Vectorized env (many copies simiultaneously) improves sample efficiency and speed
            vec_env = None
            if "CartPole" in env_name:
                vec_env = DummyVecEnv([lambda: CartPoleDanceWrapper(gym.make(env_name))])
            else:
                vec_env = DummyVecEnv([lambda: gym.make(env_name)]) 

            # Check for GPU availability and use it
            device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"

            if run_id in training_model_devices:
                device = training_model_devices[run_id]

            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                device=device,
                tensorboard_log="./tensorboard_logs"  # optional: for better training monitoring
            )
            
            # Train
            callback = ProgressBarCallback(total_timesteps=train_steps, runId=run_id)
            global train_run_progress
            train_run_progress= 0
            start_training(model, run_id=run_id, train_steps=train_steps, reset_num_timesteps=False, callback=callback)
            #model.learn(total_timesteps=train_steps, reset_num_timesteps=False, callback=callback)

            # Save
            model.save("ppo_model")

            # Optional: delete vec_env to free RAM/GPU
            vec_env.close()
        obs, _ = env.reset()
        step = 0
        episodes_seen = 0

        ep_reward = 0
        send_frame_interval = 5
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
                await asyncio.sleep(0.1)
            if train_mode:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to("cpu")
                with torch.no_grad():
                    action, _ = model.predict(obs_tensor)
                    #print("example action output: ", action)
                    if isinstance(env.action_space, gym.spaces.Discrete):
                        action = int(np.asarray(action).reshape(-1)[0])
                    else:
                        action = action.squeeze(0)
                    
            else:
                if run_id not in current_model or current_model[run_id] is None:
                    action = env.action_space.sample()
                else:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to("cpu")
                    #print("--- Using custom model right now ---")
                    with torch.no_grad():
                        action, _ = current_model[run_id].predict(obs_tensor)
                        if isinstance(env.action_space, gym.spaces.Discrete):
                            action = int(np.asarray(action).reshape(-1)[0])
                        else:
                            action = action.squeeze(0)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

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
                    #"done": done
                }

                # Send JSON over WebSocket
                await websocket.send_text(json.dumps(data))
                obs, _ = env.reset()
                step = 0
                ep_reward = 0
                episodes_seen += 1
                ep_frames = []
            else:
                if episodes_seen % send_frame_interval == 0 :
                    ep_frames.append(render_env(env))
                obs = next_obs
                step += 1
                ep_reward += reward

            await asyncio.sleep(0.05)  # throttle to ~20 FPS
    except WebSocketDisconnect:
        print("Client disconnected. ")
