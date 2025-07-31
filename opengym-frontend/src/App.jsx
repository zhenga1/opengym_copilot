import { useState, useEffect, useRef} from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import {Line} from 'react-chartjs-2'
import {Chart as ChartJS, LineElement, CategoryScale, LinearScale, PointElement} from 'chart.js'

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement);

function App() {
  const [rollouts, setRollouts] = useState([]);
  const [envName, setEnvName] = useState("CartPole-v1");
  const [isPaused, setIsPaused] = useState(false);
  const isPausedRef = useRef(false);

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
    const ws = new WebSocket('ws://localhost:8000/ws/rollout?env=${envName}');

    console.log("envName: ", envName);
    ws.onmessage = (event) => {
      /* DO NOT UPDATE THE STATE IF THE SIMULATION IS PAUSED*/
      if (isPausedRef.current) return;

      const data = JSON.parse(event.data);
      setRollouts((prev) => [data, ...prev.slice(0, 19)]);
    };

    ws.onerror = (err) => console.error("WebSocket Error: ", err);
    ws.onclose = () => console.log("WebSocket Closed. ");
    return () => ws.close();
  }, [envName]);

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
