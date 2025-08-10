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
  const removeRollout = () => {
    setWindows(prev => {
      if (prev.length === 0) return prev;           // nothing to remove
      const next = prev.slice(0, -1);               // drop the last window
      setCurrentIndex(next.length ? next.length - 1 : 0); // clamp index
      return next;
    });
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
          onClick={removeRollout}
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
          ➖ Remove Rollout
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
            {windows.map((win, i) => (
              <div
                key={i}
                style={{
                  display: i === currentIndex ? 'block' : 'none',
                }}
              >
                {win}
              </div>
            ))}
        
        </div>
      ) : (
        <p>No rollouts yet. Click "Add Rollout" to begin.</p>
      )}
    </div>
  );
}

export default App;
