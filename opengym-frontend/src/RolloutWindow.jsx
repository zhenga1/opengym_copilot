import { useState, useEffect, useRef, use} from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import {Line} from 'react-chartjs-2'
import SetPathPopup from './SetPathPopup'
import ProgressBar from './ProgressBar'
import axios  from 'axios'
import {Chart as ChartJS, LineElement, CategoryScale, LinearScale, PointElement} from 'chart.js'

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement);

function RolloutWindow() {
  const [rollouts, setRollouts] = useState([]);
  const [frames, setFrames] = useState([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  // defined
  const [renewFrameInterval, setRenewFrameInterval] = useState(5);
  const [replayInterval, setReplayInterval] = useState(50); // in ms
  const [episodeNumForSimulation, setEpisodeNumForSimulation] = useState(0);
  // whether to train in the backend
  const [trainSteps, setTrainSteps] = useState(1000);
  const [trainMode, setTrainMode] = useState(false);
  // Stores info on the CURRENT episode
  const [episodeInfo, setEpisodeInfo] = useState({episode: 0, reward: 0});

  const [envName, setEnvName] = useState("CartPole-v1");
  const [isPaused, setIsPaused] = useState(false);
  const isPausedRef = useRef(false);
  const [sessionId, setSessionId] = useState(null);
  const [runId, setRunId] = useState(null);
  const intervalRef = useRef(null);

  const socketRef = useRef(null);
  const retryRef = useRef(null);
  function formatDate(ts) {
    const d = new Date(ts);
    const yyyy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const dd = String(d.getDate()).padStart(2, "0");
    const hh = String(d.getHours()).padStart(2, "0");
    const min = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    return `${yyyy}${mm}${dd}_${hh}${min}${ss}`;
  }
  const timestamp = formatDate(Date.now());

  // Upload the files logistics:
  const [serverModels, setServerModels] = useState([]);
  const [selectedServerModel, setSelectedServerModel] = useState(""); // "" = None
  const [showPopup, setShowPopup] = useState(false);
  const [file, setFile] = useState(null);
  // whether is using default policy or not
  const [isUsingNone, setIsUsingNone] = useState(true);
  const [loading, setLoading] = useState(false);

  //set whether model parent directory file path is copied
  const [filePathCopied, setFilePathCopied] = useState(false);
  const [hoverOnFilePathButton, setHoverOnFilePathButton] = useState(false);
  const [hoverOnDeleteAllTemp, setHoverOnDeleteAllTempButton] = useState(false);

  const handleEnvChange = (e) => {
    setEnvName(e.target.value)
  }
  // This effectively flips the showPopup
  const togglePopup = () => {
    setShowPopup((prev) => !prev);
  };

  const [showPathPopup, setShowPathPopup] = useState(false);
  const [trainingPath, setTrainingPath] = useState("models/basic_model.zip");

  const openPathPopup = () => setShowPathPopup(true);
  const closePathPopup = () => setShowPathPopup(false);

  const [frozenPath, setFrozenPath] = useState(null);

  //gets the frozen Path, so the default Path doesn't change too drastically. 
  useEffect(() => {
      if (showPathPopup && frozenPath === null) {
        setFrozenPath(`models/ppo_model_${envName}_${timestamp}.zip`);
      }
      if (!showPathPopup) {
        // reset so a new one is generated next time
        setFrozenPath(null);
      }
  }, [showPathPopup, envName, frozenPath]);

  const toggleTrainPauseTogether = () => {
    const newValue = !trainMode;
    const newPauseValue = trainMode;
    setTrainMode(newValue);
    console.log("Toggling the pause value to:", newPauseValue);
    togglePause(newPauseValue); // pause if train mode is toggled
  };
  const saveTrainingPath = async (path, device) => {
    if (!runId) {
      console.warn("Run ID not set yet, cannot pause/resume");
      return;
    }
    try {
      // Set Train path FIRST
      // THEN SET THE TRAIN MODE AND toggle pause
      // persist to backend (example endpoint)
      await axios.post("/set_training_dir", { "run_id": runId, "train_dir_path": path, "device": device });
      setTrainingPath(path);
      closePathPopup();

      toggleTrainPauseTogether();

      
    } catch (e) {
      console.error("Failed to set training path:", e);
      // optionally show a toast here
    }
  };
  const deleteAllTempModels = () => {
    
  }
  const togglePause = async(ns) => {
    if (!sessionId) {
      console.warn("Session ID not set yet, cannot pause/resume");
      return;
    }
    const newState = ns !== undefined ? ns : !isPaused;
    console.log("Sending pause state:", newState);
    await axios.post("/pause_rollout", { session_id: sessionId, paused: newState });
    setIsPaused((prev) => {
      isPausedRef.current = !prev;
      return !prev;
    });
  };

  useEffect(() => {
    let retryTimeout;
    console.log("Fetching models from server...");
    const fetchModels = async () => {
      try {
        const res = await axios.get("/models");
        // Get the models that currently exist
        console.log("Available models: ", res);
        setServerModels(res.data.models || []);
      } catch (e) {
        console.error("List the models process has failed: Will retry in 5 seconds");
        //retry timeout = 5 seconds
        retryTimeout = setTimeout(fetchModels, 5000);
      }
    };
    fetchModels();
    return () => {
      if (retryTimeout) clearTimeout(retryTimeout);
    }
  }, []);
  useEffect(() => {
    async function fetchRunId() {
      const returnData = await axios.get("/unique_run_id");
      setRunId(returnData.data.run_id);
    }
    fetchRunId();
    console.log("Run id current: ", runId);
  }, []);
  /* Here we are adding envName to the dependency array of useEffect, so useEffect will rerun when envName changes*/
  useEffect(() => {

    let isActive = true;
    if (!runId) {
      console.warn("Run ID not set yet, cannot connect");
      return;
    }
    else {
      const connect = () => {
        const url = `ws://localhost:8000/ws/rollout?runid=${runId}&env=${envName}&train=${trainMode}&train_steps=${trainSteps}`
        console.log("Attempting to connect to : ", url);
        const ws = new WebSocket(url);

        ws.onopen = () => {
          console.log("[WebSocket] Connected ✅");
          socketRef.current = ws;
          retryRef.current = null;
        };

        ws.onmessage = (event) => {
          if (!isActive) return;
          /* DO NOT UPDATE THE STATE IF THE SIMULATION IS PAUSED*/
          if (isPausedRef.current) return;

          const data = JSON.parse(event.data);
          if (data.type === "session"){
            setSessionId(data.session_id);
            // the websocket does not need to record any more data
            return; 
          }
          // if data.type is not session
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
        ws.onerror = (err) => console.error("WebSocket Error: ", err);
        ws.onclose = () => {
          console.log("[WebSocket] Disconnected ❌");
          console.log("WebSocket is Active: ", isActive);
          if(!isActive) return;
          
          console.log("WebSocket Disconnected, retrying in 1s. ");
          retryRef.current = setTimeout(connect, 1000); // retry after 1 second
        }
      }
      
      setRollouts([]); // restart the graph simulation from the beginning, upon new simulation
      // initial attempt
      togglePause(false);
      connect();
      console.log("envName: ", envName);
      console.log("trainMode: ", trainMode);
      

      console.log("frames: ", frames &&frames.length)
      return () => {
        // cleanup function, run before next component runs
        isActive = false;
        if (socketRef.current) socketRef.current.close();
        if (retryRef.current) clearTimeout(retryRef.current);
      };
    }
  }, [envName,trainMode, runId]);

  // This is the useEffect for the frame Data from the video
  useEffect(() => {
    // console.log("Frames: ", frames)
    // console.log("isPlaying: ", isPlaying)
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

  const toggleTrainMode = async () => {
    if(!trainMode) {
      openPathPopup();
      // process the trainMode variable WITHIN the popup (i.e. after popup closes)
    } else {
      closePathPopup();
      toggleTrainPauseTogether();
    }
  }
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

  const handleModelUpload = (e) => {
    const f = e.target.files?.[0] || null;
    setFile(f);
  };
  const useNone = async () => {
    setLoading(true);
    setIsUsingNone(true);
    try {
      // “Clear” the session’s model by loading none; implement either:
      // 1) a dedicated endpoint:
      // await axios.post("/unload_model", { session_id: sessionId });
      // OR 2) overload load_model with a sentinel:
      await axios.post("/load_model", { run_id: runId, model_name: "" });
    } finally {
      setLoading(false);
    }
  };

  const loadServerModel = async () => {
    if (!runId) {
      console.warn("Run ID not set yet, cannot load model");
      return;
    }
    setIsUsingNone(false);
    if (!selectedServerModel) return useNone();
    setLoading(true);
    try {
      await axios.post("/load_model", {
        runId: runId,
        model_name: selectedServerModel,
      });
    } finally {
      setLoading(false);
    }
  };

  const uploadAndLoad = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file); // field name "file" expected by backend
      const up = await axios.post("/upload_model", form);
      const modelName = up.data?.model_name; // backend should return stored filename
      if (modelName) {
        await axios.post("/load_model", { runId: runId, model_name: modelName });
      }
    } finally {
      setLoading(false);
      setFile(null);
    }
  };

  const getRootSavedModelsLink = async() => {
    const result = await axios.get("/get_model_path");
    console.log("Root saved models link from backend:", result.data);
    let path = result.data;
    try {
      await navigator.clipboard.writeText(path);
      setFilePathCopied(true);
      setTimeout(() => setFilePathCopied(false), 1500); // reset after 1.5s
    } catch (err) {
      console.error("Failed to copy: ", err);
    }
  }
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

    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <label htmlFor="trainSteps" style={{ fontWeight: 600 }}>
        🧠 Train Steps:
      </label>

      <input
        id='trainSteps'
        type="range"
        min="1000"
        max="100000"
        step="1000"
        value={trainSteps}
        onChange={(e) => setTrainSteps(Number(e.target.value))}
        style={{ width: '200px', margin: '0.5rem' }}
      /> 
      <input
        type="number"
        min="1000"
        max="100000"
        step="1000"
        value={trainSteps}
        onChange={(e) =>  setTrainSteps(Number(e.target.value))}
        style={{
          width: '70px',
          padding: '4px',
          border: '1px solid #d1d5db',
          borderRadius: '4px',
        }}
      /> 
    </div>
    
    <div style={{ display: 'grid', gap: '0.75rem', margin: '1rem 0' }}>
      {/* Row: “Load Model” label + file picker */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', justifyContent: 'center' }}>
        <label htmlFor='loadModel' style={{ fontWeight: 600 }}>Load Model:</label>
        <input
          id='loadModel'
          type="file"
          accept=".zip"
          onChange={handleModelUpload}
          style={{ maxWidth: 260 }}
        />
        <button
          onClick={uploadAndLoad}
          disabled={!file || loading}
          style={{ padding: '.4rem .75rem' }}
          title="Upload selected .zip and load into this rollout session"
        >
          {loading ? "Uploading..." : "Upload & Load"}
        </button>
      </div>

      {/* Row: server model dropdown */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', justifyContent: 'center' }}>
        <label htmlFor="serverModel" style={{ fontWeight: 600 }}>From Server:</label>
        <button
        onClick={getRootSavedModelsLink}
        onMouseEnter={() => setHoverOnFilePathButton(true)}
        onMouseLeave={() => setHoverOnFilePathButton(false)}
        style={{
          border: "none",
          background: "transparent",
          cursor: "pointer",
          fontSize: "1rem", // small text-sized
          padding: "0.2rem",
        }}
        title={filePathCopied ? "Copied!" : "Copy folder link"}
      >
        {hoverOnFilePathButton ? "🔗" : "📁"}
      </button>
      {/* Short notification */}
      {filePathCopied && (
        <div
          style={{
            position: "absolute",
            top: "-1.5rem",
            left: "50%",
            transform: "translateX(-50%)",
            background: "#333",
            color: "#fff",
            fontSize: "0.75rem",
            padding: "2px 6px",
            borderRadius: "4px",
            whiteSpace: "nowrap",
          }}
        >
          Copied!
        </div>
      )}
        <select
          id="serverModel"
          value={selectedServerModel}
          onChange={(e) => setSelectedServerModel(e.target.value)}
          style={{ padding: '.35rem .5rem', minWidth: 260 }}
        >
          <option value="">(None — random rollout)</option>
          {serverModels.map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <button
          onClick={deleteAllTempModels}
          onMouseEnter={() => setHoverOnDeleteAllTempButton(true)}
          onMouseLeave={() => setHoverOnDeleteAllTempButton(false)}
          style={{
            border: "none",
            background: "transparent",
            cursor: "pointer",
            fontSize: "1rem", // small text-sized
            padding: "0.2rem",
          }}
          title={filePathCopied ? "Copied!" : "Copy folder link"}
        >
          {hoverOnDeleteAllTemp ? "🗑 Delete All" : "🗑"}
        </button>
        <button
          onClick={loadServerModel}
          disabled={loading}
          style={{ padding: '.4rem .75rem',
            backgroundColor: !isUsingNone ? '#d1d5db' : '#10b981'
          }}

          title="Load the selected server model (or None)"
        >
          {loading ? "Loading..." : "Use Selection"}
        </button>
        <button
          onClick={useNone}
          disabled={loading}
          style={{ padding: '.4rem .75rem', 
            backgroundColor: isUsingNone ? '#d1d5db' : '#10b981'
          }}
          title="Clear model for this session (random rollout)"
        >
          Use None
        </button>
      </div>
    </div>
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
        onClick={toggleTrainMode}
        style={{
          padding: '0.5rem 1.2rem',
          fontSize: '1rem',
          backgroundColor: trainMode ? '#3b82f6' : '#9ca3af', // blue if on, gray if off
          color: 'white',
          border: 'none',
          borderRadius: '999px', // pill shape
          cursor: 'pointer',
          boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
          transition: 'all 0.3s ease-in-out',
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.5rem',
        }}
      >
        {trainMode ? '🧠 Training Active' : '🚫 Training Disabled'}
      </button>
      <SetPathPopup
        isOpen={showPathPopup}
        defaultPath={frozenPath !== null ? frozenPath : `models/ppo_model_${envName}_${timestamp}.zip`}
        onConfirm={saveTrainingPath}
        onClose={closePathPopup}
      />
      
      <button
        onClick={() => togglePause()}
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
        {/* Classic Control Environments */}
        <option value="CartPole-v0">CartPole-v0</option>
        <option value="CartPole-v1">CartPole-v1</option>
        <option value="MountainCar-v0">MountainCar-v0</option>
        <option value="MountainCarContinuous-v0">MountainCarContinuous-v0</option>
        <option value="Acrobot-v1">Acrobot-v1</option>
        <option value="Pendulum-v1">Pendulum-v1</option>

        {/* Box2D */}
        <option value="LunarLander-v2">LunarLander-v2</option>
        <option value="LunarLanderContinuous-v2">LunarLanderContinuous-v2</option>
        <option value="BipedalWalker-v3">BipedalWalker-v3</option>
        <option value="BipedalWalkerHardcore-v3">BipedalWalkerHardcore-v3</option>
        <option value="CarRacing-v3">CarRacing-v3</option>

        {/*Mujoco (Continuous Control)*/}
        <option value="Humanoid-v4">Humanoid-v4</option>
        <option value="HumanoidStandup-v4">HumanoidStandup-v4</option>
        <option value="Ant-v4">Ant-v4</option>
        <option value="HalfCheetah-v4">HalfCheetah-v4</option>
        <option value="Hopper-v4">Hopper-v4</option>
        <option value="Walker2d-v4">Walker2d-v4</option>
        <option value="Swimmer-v4">Swimmer-v4</option>
        <option value="Reacher-v4">Reacher-v4</option>
        <option value="Pusher-v4">Pusher-v4</option>
        <option value="InvertedPendulum-v4">InvertedPendulum-v4</option>
        <option value="InvertedDoublePendulum-v4">InvertedDoublePendulum-v4</option>
      </select>
    </div>
      
    {trainMode && (
      <ProgressBar isTraining={trainMode} />
    // <div style={{ margin: '1.5rem auto', textAlign: 'center' }}>
    //   <div style={{
    //     height: '8px',
    //     width: '60%',
    //     backgroundColor: '#e5e7eb',
    //     borderRadius: '999px',
    //     overflow: 'hidden',
    //     margin: '0 auto',
    //     position: 'relative'
    //   }}>
    //     <div style={{
    //       height: '100%',
    //       width: '40%',
    //       backgroundColor: '#3b82f6',
    //       animation: 'progress-slide 1.5s infinite ease-in-out'
    //     }} />
    //   </div>
    //   <p style={{ marginTop: '0.5rem', color: '#4b5563', fontWeight: 500 }}>
    //     Training in progress...
    //   </p>
    // </div>
  )}


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
    {/*<RolloutSlideshow/>*/}
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

    <div style={{ marginTop: '3rem' }}>
      <h3 style={{ fontSize: '1.25rem', color: '#6366f1' }}>📈 Reward Chart</h3>
      <p>
        Episode <strong>{episodeInfo.episode}</strong>, Reward:{' '}
        <strong style={{ color: '#10b981' }}>{episodeInfo.reward}</strong>
      </p>
      <div style={{ width: '100%', maxWidth: '600px', height: '300px', margin: '0 auto'}}>
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

export default RolloutWindow
