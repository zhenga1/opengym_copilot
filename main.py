from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import gymnasium as gym
import json

app = FastAPI()

@app.websocket("/ws/rollout")
async def rollout_stream(websocket: WebSocket, env_name:str = "CartPole-v1"):
    await websocket.accept()

    # get a query parameter
    #query = websocket.headers.get("sec-websocket-protocol", "CartPole-v1")
    #env_name = query or "CartPole-v1"
    

    try:
        env = gym.make(env_name, render_mode="rgb_array")
        obs, _ = env.reset()
        step = 0

        while True:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Construct the data payload
            data = {
                "step": step,
                "observation": obs.tolist(),
                "action": int(action),
                "reward": float(reward),
                "done": done
            }

            # Send JSON over WebSocket
            await websocket.send_text(json.dumps(data))

            # Prepare for next step
            if done:
                obs, _ = env.reset()
                step = 0
            else:
                obs = next_obs
                step += 1

            await asyncio.sleep(0.05)  # throttle to ~20 FPS
    except WebSocketDisconnect:
        print("Client disconnected. ")
