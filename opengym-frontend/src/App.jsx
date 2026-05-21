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
  savedRewardConfigFiles: [],
  selectedRewardConfigFile: '',
  rewardConfigSaveName: '',
  rewardConfigSaveSourceType: 'manual',
  taskGoal: '',
  taskProposal: null,
  taskProposalLoading: false,
  taskProposalStatus: 'Describe a task goal, then generate a proposal.',
  taskProposalLiveStatus: null,
  latestTrainingBreakdown: {},
  latestTrainingMeanBreakdown: {},
  latestRolloutBreakdown: {},
  rewardLogs: [],
  onTermChange: () => {},
  onAddCustomTerm: () => {},
  onRemoveTerm: () => {},
  onSaveConfig: () => {},
  onTaskGoalChange: () => {},
  onProposeTaskConfig: () => {},
  onApplyTaskProposal: () => {},
  onRewardConfigFileSelect: () => {},
  onRewardConfigSaveNameChange: () => {},
  onRewardConfigSaveSourceTypeChange: () => {},
  onSaveRewardConfigSnapshot: () => {},
  onLoadRewardConfigSnapshot: () => {},
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
    <div className="app-shell">
      <header className="app-topbar">
        <div className="app-brand-block">
          <div className="app-kicker">Reward Engineering Workbench</div>
          <h1 className="app-title">OpenGym Copilot</h1>
          <p className="app-subtitle">
            Tune rewards, compare policies, and inspect failures with a cleaner training and rollout workflow.
          </p>
        </div>
        <div className="app-toolbar">
          <button className="nav-button" onClick={prev} disabled={currentIndex === 0}>
            <span>&#8592;</span>
          </button>
          <button onClick={addRollout} className="app-action-button primary">
            Add Rollout
          </button>
          <button onClick={removeRollout} className="app-action-button ghost">
            Remove Rollout
          </button>
          <button className="nav-button" onClick={next} disabled={currentIndex >= windows.length - 1}>
            <span>&#8594;</span>
          </button>
        </div>
      </header>

      {windows.length > 0 ? (
        <div className="app-workspace">
          {windows.map((win, i) => (
            <div
              key={win.id}
              style={{
                display: i === currentIndex ? 'block' : 'none',
                padding: '0.25rem',
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
        <div className="app-empty-state">
          <div className="app-empty-card">
            <div className="app-empty-title">No rollouts yet</div>
            <div className="app-empty-copy">
              Start a rollout window to train a policy, inspect reward structure, and compare saved experiments.
            </div>
            <button onClick={addRollout} className="app-action-button primary">
              Add First Rollout
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
