import { useState } from 'react';
import RolloutWindow from './RolloutWindow';
import './App.css';

function App() {
  const [windows, setWindows] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  const addRollout = () => {
    const newWindow = <RolloutWindow key={windows.length} />;
    setWindows((prev) => [...prev, newWindow]);
    setCurrentIndex(windows.length); // slide to the new one
  };

  const prev = () => {
    if (currentIndex > 0) setCurrentIndex((i) => i - 1);
  };

  const next = () => {
    if (currentIndex < windows.length - 1) setCurrentIndex((i) => i + 1);
  };

  return (
    <div style={{ padding: '1rem', textAlign: 'center' }}>
      <div style={{ marginBottom: '1rem' }}>
        <button onClick={prev} disabled={currentIndex === 0}>⬅️</button>

        <button
          onClick={addRollout}
          style={{
            margin: '0 1rem',
            backgroundColor: '#8b5cf6',
            color: 'white',
            padding: '0.5rem 1rem',
            borderRadius: '8px',
            border: 'none',
            cursor: 'pointer',
            fontWeight: 'bold',
          }}
        >
          ➕ Add Rollout
        </button>

        <button
          onClick={next}
          disabled={currentIndex >= windows.length - 1}
        >➡️</button>
      </div>

      {windows.length > 0 ? (
        <div
            style={{
              width: '100%',
              display: 'flex',
              justifyContent: 'center',
            }}
          >
            <div
              style={{
                width: '100%',
                maxWidth: '800px',
                margin: '0 auto',
                textAlign: 'center',
              }}
            >
            {windows[currentIndex]}
          </div>
        
        </div>
      ) : (
        <p>No rollouts yet. Click "Add Rollout" to begin.</p>
      )}
    </div>
  );
}

export default App;
