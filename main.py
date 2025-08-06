from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import gymnasium as gym
import torch
from stable_baselines3 import PPO
import json
import cv2
import base64
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
#import constants

from urllib.parse import parse_qs

app = FastAPI()
env_name = "" # unknown for now

device = "cuda" if torch.cuda.is_available() else "cpu"

file = open("progress_bar.log", "w")
train_run_progress = 0
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
    
    def _on_step(self) -> bool:
        global train_run_progress
        # bound the progress bar percentage by the minimum of n_calls and total_timesteps, so self.n_calls never exceeds self.total_timesteps
        pct = 100 * min(self.n_calls, self.total_timesteps) / self.total_timesteps
        # file.write(f"Progress: {pct:.2f}%\n")
        # file.write(f"n_calls: {self.n_calls}\n total_timesteps: {self.total_timesteps}\n")
        # file.flush()
        print(f"Progress: {pct:.2f}%", end='\r') # or send to the frontend
        
        train_run_progress = min(int(pct), 100)
        return True
    
from fastapi import WebSocket
from threading import Thread
from datetime import datetime
#@app.post("/start")
def start_training(model: PPO = None, train_steps:int = 1000, reset_num_timesteps = False,callback: BaseCallback = None):
    def train():
        global train_run_progress
        train_run_progress = 0
        model.learn(total_timesteps=train_steps, reset_num_timesteps=False, callback=callback)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model.save(f"models/ppo_model_{env_name}_{timestamp}.zip")
        print("Training complete")
    Thread(target=train).start()
    # status option
    return {"status": "training started"}

@app.get("/progress")
def get_progress():
#async def websocket_endpoint(websocket: WebSocket):
    # await websocket.accept()
    # for i in range(100):
    #     await websocket.send_json({"progress": train_run_progress})
    #     await asyncio.sleep(0.1)
    # await websocket.close()
    file.write(f"Put variable by name progress: {train_run_progress}\n")
    return {"progress": train_run_progress}
@app.websocket("/ws/rollout")
async def rollout_stream(websocket: WebSocket):#, env_name:str = "CartPole-v1"):
    global env_name
    await websocket.accept()

    query = parse_qs(websocket.url.query)
    print("query: ", query)
    env_name = query.get("env", ["CartPole-v1"])[0]
    train_mode_str = query.get("train", ["true"])[0]
    train_mode = train_mode_str.lower() == "true"   # ✅ real boolean
    train_steps = query.get("train_steps", [1000])[0]
    train_steps = int(train_steps)


    print("env_name: ", env_name)
    print("train_steps: ", train_steps)
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
            # Vectorized env improves sample efficiency and speed
            vec_env = DummyVecEnv([lambda: gym.make(env_name)])
            
            # Check for GPU availability and use it
            device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"

            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                device=device,
                tensorboard_log="./tensorboard_logs"  # optional: for better training monitoring
            )
            
            # Train
            callback = ProgressBarCallback(total_timesteps=train_steps)
            global train_run_progress
            train_run_progress= 0
            start_training(model, train_steps=train_steps, reset_num_timesteps=False, callback=callback)
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
            if train_mode:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to("cpu")
                with torch.no_grad():
                    action, _ = model.predict(obs_tensor)
                    #print("example action output: ", action)
                    if isinstance(env.action_space, gym.spaces.Discrete):
                        action = action.item()
                    else:
                        action = action.cpu().numpy()[0]
                    
            else:
                action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            

            # Prepare for next step
            if done:
                frames = ep_frames if episodes_seen % send_frame_interval == 0 else []
                sim_frame_episode_number = episodes_seen if episodes_seen % send_frame_interval == 0 else None
                # Construct the data payload
                data = {
                    #"step": step,
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
