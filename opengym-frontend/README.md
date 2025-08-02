# 🎨 OpenGym Copilot Frontend

This is the **React-based frontend** for [OpenGym Copilot](../README.md) — a real-time rollout visualizer for reinforcement learning environments like OpenAI Gym.

It connects to a FastAPI backend and streams:
- Agent rollouts
- Episode rewards
- Frame-by-frame visualizations

---

## 🚀 Features

- 🧠 Live agent rollout viewer (WebSocket-connected)
- 🖼 Animated episode playback with frame loop
- 🕹 Environment selector
- 🎛 Dynamic control for:
  - Frame playback speed
  - Backend episode-frame interval
- ⏸ Pause & resume streaming
- 📈 Real-time reward charting

---

## 🛠 Tech Stack

- **React 18**
- **Vite** (fast dev server)
- **Chart.js** for reward plotting
- **WebSocket** for rollout streaming

---

## ⚙️ Setup

### 1. Install dependencies

```bash
npm install
```

### 2. Run the application!
```bash
npm run dev
```