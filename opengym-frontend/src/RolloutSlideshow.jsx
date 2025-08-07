import { useState } from 'react';
import RolloutCard from './RolloutCard';

export default function RolloutSlideshow() {
  const [rollouts, setRollouts] = useState([]);
  const [currentRolloutIndex, setCurrentRolloutIndex] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);

  const addRollout = async () => {
    useEffect(() => {
        let isActive = true;
        const connect = () => {
          const url = `ws://localhost:8000/ws/rollout?env=${envName}&train=${trainMode}&train_steps=${trainSteps}`
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
            if(!isActive) return;
            
            console.log("WebSocket Disconnected, retrying in 1s. ");
            retryRef.current = setTimeout(connect, 1000); // retry after 1 second
          }
        }
      })
  };

  const prevRollout = () => {
    if (currentRolloutIndex > 0) {
      setCurrentRolloutIndex((prev) => prev - 1);
      setCurrentFrame(0);
    }
  };

  const nextRollout = () => {
    if (currentRolloutIndex < rollouts.length - 1) {
      setCurrentRolloutIndex((prev) => prev + 1);
      setCurrentFrame(0);
    }
  };

  const currentRollout = rollouts[currentRolloutIndex];

  return (
    <div style={{ padding: '1rem' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginBottom: '1.5rem',
          alignItems: 'center',
        }}
      >
        <button
          onClick={addRollout}
          style={{
            backgroundColor: '#8b5cf6',
            color: 'white',
            padding: '0.5rem 1rem',
            borderRadius: '8px',
            border: 'none',
            cursor: 'pointer',
            fontWeight: 'bold',
          }}
        >
          📥 Add Rollout
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <button onClick={prevRollout} disabled={currentRolloutIndex === 0}>
            ⬅️
          </button>
          <span>
            Rollout {rollouts.length === 0 ? 0 : currentRolloutIndex + 1} of {rollouts.length}
          </span>
          <button
            onClick={nextRollout}
            disabled={currentRolloutIndex >= rollouts.length - 1}
          >
            ➡️
          </button>
        </div>
      </div>

      {currentRollout ? (
        <RolloutCard
          frames={currentRollout.frames}
          currentFrame={currentFrame}
          episodeInfo={{
            episode: currentRollout.episode,
            reward: currentRollout.reward,
          }}
          rollouts={rollouts}
        />
      ) : (
        <p style={{ textAlign: 'center', color: '#666' }}>
          No rollouts yet. Click “Add Rollout” to begin.
        </p>
      )}
    </div>
  );
}
