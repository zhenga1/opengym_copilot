import { useState } from 'react';
import RolloutWindow from './RolloutWindow';
import RewardSidebar from './RewardSidebar';
import './App.css';

const emptySidebarState = {
  envName: 'No rollout selected',
  rewardConfig: [],
  rewardConfigDirty: false,
  rewardConfigLoading: false,
  rewardConfigStatus: 'Create or select a rollout to inspect reward terms.',
  supportsCustomReward: true,
  availableRewardVariables: [],
  rewardFormulaExamples: [],
  rewardSourceLinks: [],
  latestTrainingBreakdown: {},
  latestTrainingMeanBreakdown: {},
  latestRolloutBreakdown: {},
  rewardLogs: [],
  onTermChange: () => {},
  onAddCustomTerm: () => {},
  onRemoveTerm: () => {},
  onSaveConfig: () => {},
};

function App() {
  const [windows, setWindows] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [sidebarState, setSidebarState] = useState(emptySidebarState);

  const addRollout = () => {
    const newWindow = { id: crypto.randomUUID(), mode: 'live' };
    setWindows((prev) => {
      const next = [...prev, newWindow];
      setCurrentIndex(next.length - 1);
      return next;
    });
  };

  const addLoadedRollout = ({ rollouts, envName, fileName }) => {
    const newWindow = {
      id: crypto.randomUUID(),
      mode: 'saved',
      initialRollouts: rollouts,
      initialEnvName: envName,
      viewerLabel: fileName,
    };
    setWindows((prev) => {
      const next = [...prev, newWindow];
      setCurrentIndex(next.length - 1);
      return next;
    });
  };

  const removeRollout = () => {
    setWindows((prev) => {
      if (prev.length === 0) return prev;
      const next = prev.slice(0, -1);
      const nextIndex = next.length ? next.length - 1 : 0;
      setCurrentIndex(nextIndex);
      if (next.length === 0) {
        setSidebarState(emptySidebarState);
      }
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
    <div
      style={{
        padding: '2rem',
        fontFamily: 'Segoe UI, sans-serif',
        maxWidth: '1500px',
        margin: '0 auto',
        textAlign: 'center',
      }}
    >
      <div style={{ marginBottom: '1rem' }}>
        <button onClick={prev} disabled={currentIndex === 0}> <span>&#8592;</span> </button>

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
          Add Rollout
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
          Remove Rollout
        </button>

        <button
          onClick={next}
          disabled={currentIndex >= windows.length - 1}
        > <span>&#8594;</span> </button>
      </div>

      {windows.length > 0 ? (
        <div
          style={{
            width: '100%',
            paddingTop: '1rem',
            paddingBottom: 16,
          }}
        >
          {windows.map((win, i) => (
            <div
              key={win.id}
              style={{
                display: i === currentIndex ? 'block' : 'none',
                padding: '1rem',
              }}
            >
              <RolloutWindow
                viewerMode={win.mode || 'live'}
                initialRollouts={win.initialRollouts || []}
                initialEnvName={win.initialEnvName || 'CartPole-v1'}
                viewerLabel={win.viewerLabel || ''}
                isActive={i === currentIndex}
                onSidebarStateChange={setSidebarState}
                onOpenLoadedRollout={addLoadedRollout}
              />
            </div>
          ))}

          <RewardSidebar {...sidebarState} />
        </div>
      ) : (
        <p>No rollouts yet. Click "Add Rollout" to begin.</p>
      )}
    </div>
  );
}

export default App;
