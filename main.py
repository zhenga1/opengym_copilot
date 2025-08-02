from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import gymnasium as gym
import json
import cv2
import base64

from urllib.parse import parse_qs

app = FastAPI()


@app.websocket("/ws/rollout")
async def rollout_stream(websocket: WebSocket):#, env_name:str = "CartPole-v1"):
    await websocket.accept()

    query = parse_qs(websocket.url.query)
    env_name = query.get("env", ["CartPole-v1"])[0]

    print("env_name: ", env_name)
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
        obs, _ = env.reset()
        step = 0
        episodes_seen = 0

        ep_reward = 0
        send_frame_interval = 5
        ep_frames = []
        while True:
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
                ep_reward += 1

            await asyncio.sleep(0.05)  # throttle to ~20 FPS
    except WebSocketDisconnect:
        print("Client disconnected. ")
