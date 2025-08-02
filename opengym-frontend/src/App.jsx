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
    <div style={{ padding: '1rem', fontFamily: 'sans-serif' }}>
      <h1>OpenGym Rollout Viewer</h1>
      <button
        onClick={togglePause}
        style={{
          padding: '0.5rem 1rem',
          marginBottom: '1rem',
          fontSize: '1rem',
          cursor: 'pointer',
        }}
      >
        {isPaused ? '▶ Continue' : '⏸ Pause'}
      </button>
      <label>Environment: </label>
      <select value={envName} onChange={handleEnvChange}>
        <option value="CartPole-v1">CartPole-v1</option>
        <option value="MountainCar-v0">MountainCar-v0</option>
        <option value="Acrobot-v1">Acrobot-v1</option>
        <option value="Humanoid-v4">Humanoid-v4</option>
      </select>
      <h3>Simulation Toggler</h3>
      <div className="mt-4 flex gap-3">
        <button onClick={handlePlay}>▶️ Play</button>
        <button onClick={handlePause}>⏸ Pause</button>
        <button onClick={handleRestart}>⏮ Restart</button>
      </div>
      <p>Simulating Episode {episodeNumForSimulation}</p>
      <div style={{ marginTop: '1rem' }}>
      <label htmlFor="replaySpeed">Playback Speed (ms per frame):</label>
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
      {frames &&frames.length > 0 && (
        <img
          src={`data:image/jpeg;base64,${frames[currentFrame]}`}
          alt={`frame ${currentFrame}`}
          className="w-full max-w-xl rounded shadow"
        />
      )}
      <h3>Reward Curve per Episode</h3>
      <p>Episode {episodeInfo.episode}, Reward: {episodeInfo.reward}. Visualization:</p>
      <ul>
        {/* {rollouts.map((r, idx) => (
          <li key={idx}>
            <strong>Step {r.step}</strong> – Action: {r.action}, Reward: {r.reward}, Done: {String(r.done)}
          </li>
        ))} */
          
          <div style={{ width: '600px', height: '300px' }}>
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
            }
      </ul>

    </div>
  );
}

export default App
