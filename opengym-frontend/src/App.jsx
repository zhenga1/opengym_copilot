import { useState, useEffect, useRef} from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import {Line} from 'react-chartjs-2'
import {Chart as ChartJS, LineElement, CategoryScale, LinearScale, PointElement} from 'chart.js'

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement);

function App() {
  const [rollouts, setRollouts] = useState([]);
  const [frames, setFrames] = useState([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  // defined
  const [renewFrameInterval, setRenewFrameInterval] = useState(5);
  const [replayInterval, setReplayInterval] = useState(50); // in ms
  const [episodeNumForSimulation, setEpisodeNumForSimulation] = useState(0);
  // Stores info on the CURRENT episode
  const [episodeInfo, setEpisodeInfo] = useState({episode: 0, reward: 0});

  const [envName, setEnvName] = useState("CartPole-v1");
  const [isPaused, setIsPaused] = useState(false);
  const isPausedRef = useRef(false);
  const intervalRef = useRef(null);

  const handleEnvChange = (e) => {
    setEnvName(e.target.value)
  }

  const togglePause = () => {
    setIsPaused((prev) => {
      isPausedRef.current = !prev;
      return !prev;
    });
  };

  /* Here we are adding envName to the dependency array of useEffect, so useEffect will rerun when envName changes*/
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/rollout?env=${envName}`);
    setRollouts([]); // restart the graph simulation from the beginning, upon new simulation
    console.log("envName: ", envName);
    ws.onmessage = (event) => {
      /* DO NOT UPDATE THE STATE IF THE SIMULATION IS PAUSED*/
      if (isPausedRef.current) return;

      const data = JSON.parse(event.data);
      setEpisodeInfo({ episode: data.episode, reward: data.reward });
      if(data.ep_frames.length > 0){
        setFrames(data.ep_frames);        // store all frames
        setCurrentFrame(0);            // start at first frame
      }
      if(data.sim_frame_episode_number) {
        setEpisodeNumForSimulation(data.sim_frame_episode_number);
      }
      //setIsPlaying(true); <- playback controlled by isPlaying var           // start playback automatically
      setRollouts((prev) => [data, ...prev.slice(0, 19)]);
    };

    console.log("frames: ", frames &&frames.length)
    ws.onerror = (err) => console.error("WebSocket Error: ", err);
    ws.onclose = () => console.log("WebSocket Closed. ");
    return () => ws.close();
  }, [envName]);

  // This is the useEffect for the frame Data from the video
  useEffect(() => {
    console.log("Frames: ", frames)
    console.log("isPlaying: ", isPlaying)
    if (!isPlaying || (frames && frames.length) === 0) return;
    //advance frame at ferquency of 20fps
    intervalRef.current = setInterval(() => {
      setCurrentFrame((prev) => {
        if (Array.isArray(frames) && prev < frames.length - 1) return prev + 1;  // advance frame
        // want to implement looping, so no stop at end
        //clearInterval(intervalRef.current);             // stop at end
        return 0;
      });
    }, replayInterval); // ~20 FPS

    return () => clearInterval(intervalRef.current); // clean up
  }, [isPlaying, frames]);

  const handlePlay = () => setIsPlaying(true);
  const handlePause = () => {
    console.log("handlePause");
    setIsPlaying(false);
    console.log("isPlaying.current: ", isPlaying);
    clearInterval(intervalRef.current);
  };
  const handleRestart = () => {
    setCurrentFrame(0);
    setIsPlaying(true);
  };

  /*console.log("Rollout rewards:", rollouts.map((r) => r.reward));*/
  return (
  <div style={{ padding: '2rem', fontFamily: 'sans-serif', maxWidth: '900px', margin: 'auto' }}>
    <h1 style={{ fontSize: '2rem', fontWeight: 'bold', textAlign: 'center' }}>🧠 OpenGym Copilot</h1>

    {/* TOP CONTROLS */}
    <div style={{ display: 'flex',justifyContent:'center', gap: '1rem', alignItems: 'center', margin: '1.5rem 0' }}>
      <button
        onClick={togglePause}
        style={{
          padding: '0.5rem 1rem',
          fontSize: '1rem',
          backgroundColor: isPaused ? '#4ade80' : '#f87171',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          cursor: 'pointer',
        }}
      >
        {isPaused ? '▶ Continue' : '⏸ Pause'}
      </button>

      <label htmlFor="envSelect">Environment:</label>
      <select
        id="envSelect"
        value={envName}
        onChange={handleEnvChange}
        style={{ padding: '4px 8px', fontSize: '1rem' }}
      >
        <option value="CartPole-v1">CartPole-v1</option>
        <option value="MountainCar-v0">MountainCar-v0</option>
        <option value="Acrobot-v1">Acrobot-v1</option>
        <option value="Humanoid-v4">Humanoid-v4</option>
      </select>
    </div>

    {/* SIMULATION CONTROLS */}
    <div style={{ marginTop: '2rem' }}>
      <h3 style={{ fontSize: '1.2rem' }}>🎮 Simulation Controls</h3>
      <div style={{ marginTop: '0.5rem',
        display: 'flex',
        justifyContent: 'center',
        gap: '1rem',
        }}>
        <button onClick={handlePlay}>▶️ Play</button>
        <button onClick={handlePause}>⏸ Pause</button>
        <button onClick={handleRestart}>⏮ Restart</button>
      </div>
      <p style={{ marginTop: '0.5rem' }}>Simulating Episode <strong>{episodeNumForSimulation}</strong></p>
    </div>

    {/* PLAYBACK SPEED */}
    <div style={{ marginTop: '1.5rem' }}>
      <label htmlFor="replaySpeed"><strong>Playback Speed</strong> (ms per frame):</label>
      <input
        id="replaySpeed"
        type="range"
        min="10"
        max="500"
        step="10"
        value={replayInterval}
        onChange={(e) => setReplayInterval(Number(e.target.value))}
        style={{ width: '200px', margin: '0 1rem' }}
      />
      <input
        type="number"
        min="10"
        max="500"
        step="10"
        value={replayInterval}
        onChange={(e) => setReplayInterval(Number(e.target.value))}
        style={{ width: '60px' }}
      />
    </div>

    {/* FRAME DISPLAY */}
    {frames && frames.length > 0 && (
      <div style={{ marginTop: '2rem', textAlign: 'center' }}>
        <img
          src={`data:image/jpeg;base64,${frames[currentFrame]}`}
          alt={`frame ${currentFrame}`}
          style={{ width: '100%', maxWidth: '600px', borderRadius: '10px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
        />
      </div>
    )}

    {/* REWARD CHART */}
    <div style={{ marginTop: '3rem' }}>
      <h3 style={{ fontSize: '1.2rem' }}>📈 Reward Curve</h3>
      <p>Episode <strong>{episodeInfo.episode}</strong>, Reward: <strong>{episodeInfo.reward}</strong></p>
      <div style={{ width: '100%', maxWidth: '600px', height: '300px' }}>
        <Line
          data={{
            labels: rollouts.map((r) => r.episode).reverse(),
            datasets: [
              {
                label: "Reward",
                data: rollouts.map((r) => r.reward).reverse(),
                fill: false,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
              },
            ],
          }}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { title: { display: true, text: "Episode" } },
              y: { title: { display: true, text: "Reward" } },
            },
          }}
        />
      </div>
    </div>
  </div>
);

}

export default App
