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

  const buttonStyle = (bg) => ({
    padding: '0.4rem 1rem',
    backgroundColor: bg,
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    fontWeight: 600,
    fontSize: '1rem',
    cursor: 'pointer',
    boxShadow: '0 3px 6px rgba(0,0,0,0.15)',
    transition: 'all 0.2s ease',
  });

  /*console.log("Rollout rewards:", rollouts.map((r) => r.reward));*/
  return (
  <div
    style={{
      padding: '2rem',
      fontFamily: 'Segoe UI, sans-serif',
      maxWidth: '900px',
      margin: 'auto',
      background: 'linear-gradient(145deg, #f0f9ff, #e0e7ff)',
      borderRadius: '12px',
      boxShadow: '0 8px 20px rgba(0,0,0,0.1)',
    }}
  >
    <h1 style={{ fontSize: '2.2rem', fontWeight: 700, textAlign: 'center', color: '#4f46e5' }}>
      ⚡ OpenGym Copilot
    </h1>

    {/* Top Control Row */}
    <div
      style={{
        display: 'flex',
        gap: '1rem',
        alignItems: 'center',
        margin: '2rem 0 1rem 0',
        justifyContent: 'center',
      }}
    >
      <button
        onClick={togglePause}
        style={{
          padding: '0.5rem 1.2rem',
          fontSize: '1rem',
          backgroundColor: isPaused ? '#10b981' : '#ef4444',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          cursor: 'pointer',
          boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
        }}
      >
        {isPaused ? '▶ Continue' : '⏸ Pause'}
      </button>

      <label htmlFor="envSelect" style={{ fontWeight: 600 }}>Environment:</label>
      <select
        id="envSelect"
        value={envName}
        onChange={handleEnvChange}
        style={{
          padding: '6px 10px',
          borderRadius: '6px',
          border: '1px solid #cbd5e1',
          backgroundColor: '#f9fafb',
          fontSize: '1rem',
        }}
      >
        <option value="CartPole-v1">CartPole-v1</option>
        <option value="MountainCar-v0">MountainCar-v0</option>
        <option value="Acrobot-v1">Acrobot-v1</option>
        <option value="Humanoid-v4">Humanoid-v4</option>
      </select>
    </div>

    {/* Playback Controls */}
    <div style={{ marginTop: '2rem', textAlign: 'center' }}>
      <h3 style={{ fontSize: '1.25rem', marginBottom: '1rem', color: '#3b82f6' }}>🎮 Simulation Controls</h3>
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '1rem',
          marginBottom: '0.5rem',
        }}
      >
        <button style={buttonStyle('#3b82f6')} onClick={handlePlay}>▶️ Play</button>
        <button style={buttonStyle('#8b5cf6')} onClick={handlePause}>⏸ Pause</button>
        <button style={buttonStyle('#f97316')} onClick={handleRestart}>⏮ Restart</button>
      </div>
      <p style={{ fontSize: '1rem' }}>
        Simulating Episode <strong style={{ color: '#0ea5e9' }}>{episodeNumForSimulation}</strong>
      </p>
    </div>

    {/* Playback Speed Slider */}
    <div style={{ marginTop: '2rem', textAlign: 'center' }}>
      <label htmlFor="replaySpeed" style={{ fontWeight: 600 }}>
        🎞 Frame Playback Speed:
      </label>
      <br />
      <input
        id="replaySpeed"
        type="range"
        min="10"
        max="500"
        step="10"
        value={replayInterval}
        onChange={(e) => setReplayInterval(Number(e.target.value))}
        style={{ width: '200px', margin: '0.5rem' }}
      />
      <input
        type="number"
        min="10"
        max="500"
        step="10"
        value={replayInterval}
        onChange={(e) => setReplayInterval(Number(e.target.value))}
        style={{
          width: '70px',
          padding: '4px',
          border: '1px solid #d1d5db',
          borderRadius: '4px',
        }}
      />
    </div>

    {/* Frame display */}
    {frames && frames.length > 0 && (
      <div style={{ marginTop: '2rem', textAlign: 'center' }}>
        <img
          src={`data:image/jpeg;base64,${frames[currentFrame]}`}
          alt={`frame ${currentFrame}`}
          style={{
            width: '100%',
            maxWidth: '600px',
            borderRadius: '12px',
            boxShadow: '0 4px 16px rgba(0,0,0,0.15)',
            border: '3px solid #c084fc',
          }}
        />
      </div>
    )}

    {/* Reward Chart */}
    <div style={{ marginTop: '3rem' }}>
      <h3 style={{ fontSize: '1.25rem', color: '#6366f1' }}>📈 Reward Chart</h3>
      <p>
        Episode <strong>{episodeInfo.episode}</strong>, Reward:{' '}
        <strong style={{ color: '#10b981' }}>{episodeInfo.reward}</strong>
      </p>
      <div style={{ width: '100%', maxWidth: '600px', height: '300px' }}>
        <Line
          data={{
            labels: rollouts.map((r) => r.episode).reverse(),
            datasets: [
              {
                label: "Reward",
                data: rollouts.map((r) => r.reward).reverse(),
                fill: false,
                borderColor: 'rgb(56, 189, 248)',
                backgroundColor: 'rgba(56, 189, 248, 0.2)',
                tension: 0.25,
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
