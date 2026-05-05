import { useEffect, useMemo, useRef, useState } from 'react';

function RewardSidebar({
  envName,
  rewardConfig,
  rewardConfigDirty,
  rewardConfigLoading,
  rewardConfigStatus,
  supportsCustomReward,
  latestTrainingBreakdown,
  latestTrainingMeanBreakdown,
  latestRolloutBreakdown,
  rewardLogs,
  onTermChange,
  onSaveConfig,
}) {
  const safeRewardConfig = Array.isArray(rewardConfig) ? rewardConfig : [];
  const safeRewardLogs = Array.isArray(rewardLogs) ? rewardLogs : [];
  const rolloutEntries = Object.entries(latestRolloutBreakdown || {}).filter(([key, value]) => key !== 'total' && Number.isFinite(value));
  const trainingEntries = Object.entries(latestTrainingBreakdown || {}).filter(([key, value]) => key !== 'total' && Number.isFinite(value));
  const fallbackEntries = rolloutEntries.length > 0 ? rolloutEntries : trainingEntries;
  const hasOnlyNativeConfig =
    safeRewardConfig.length === 0 ||
    (safeRewardConfig.length === 1 && safeRewardConfig[0]?.key === 'native');

  const editableTerms = hasOnlyNativeConfig
    ? fallbackEntries.map(([key]) => {
        const existing = safeRewardConfig.find((term) => term.key === key);
        return existing ?? {
          key,
          label: key.replaceAll('_', ' ').replace(/\b\w/g, (char) => char.toUpperCase()),
          description: 'Recovered from live reward breakdown. Change the weight and apply to persist it.',
          enabled: true,
          weight: key === 'native' ? 1 : 0,
        };
      })
    : safeRewardConfig;

  const canTuneFromBreakdown = editableTerms.length > 0;
  const sidebarSupportsCustomReward = supportsCustomReward || editableTerms.length > 1;

  const [panelPosition, setPanelPosition] = useState({ x: 24, y: 110 });
  const [isDragging, setIsDragging] = useState(false);
  const dragOffsetRef = useRef({ x: 0, y: 0 });
  const [openSections, setOpenSections] = useState({
    rewardTerms: true,
    latestTraining: true,
    trainingMean: false,
    latestRollout: true,
    logs: false,
  });
  const [expandedTerms, setExpandedTerms] = useState({});

  const visibleTermKeys = useMemo(
    () => editableTerms.map((term) => term.key),
    [editableTerms]
  );

  useEffect(() => {
    setExpandedTerms((prev) => {
      const next = {};
      for (const key of visibleTermKeys) {
        next[key] = prev[key] ?? false;
      }
      return next;
    });
  }, [visibleTermKeys]);

  useEffect(() => {
    if (!isDragging) return undefined;

    const onMouseMove = (event) => {
      setPanelPosition({
        x: Math.max(8, event.clientX - dragOffsetRef.current.x),
        y: Math.max(8, event.clientY - dragOffsetRef.current.y),
      });
    };

    const onMouseUp = () => {
      setIsDragging(false);
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);

    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [isDragging]);

  const startDrag = (event) => {
    const rect = event.currentTarget.parentElement?.getBoundingClientRect();
    if (!rect) return;
    dragOffsetRef.current = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
    setIsDragging(true);
  };

  const toggleSection = (key) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const toggleTerm = (key) => {
    setExpandedTerms((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const panelStyle = {
    width: '360px',
    maxHeight: 'calc(100vh - 32px)',
    overflowY: 'auto',
    background: 'linear-gradient(180deg, #0f172a, #111827)',
    color: '#e5eefb',
    borderRadius: '16px',
    padding: '1rem',
    boxShadow: '0 20px 44px rgba(15, 23, 42, 0.34)',
    position: 'fixed',
    left: `${panelPosition.x}px`,
    top: `${panelPosition.y}px`,
    zIndex: 50,
    userSelect: isDragging ? 'none' : 'auto',
    border: '1px solid rgba(148, 163, 184, 0.18)',
  };

  const cardStyle = {
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
    border: '1px solid rgba(148, 163, 184, 0.18)',
    borderRadius: '12px',
    padding: '0.85rem',
    marginTop: '0.9rem',
  };

  const sectionHeaderStyle = {
    width: '100%',
    background: 'transparent',
    border: 'none',
    color: '#e5eefb',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 0,
    cursor: 'pointer',
    fontSize: '0.95rem',
    fontWeight: 700,
  };

  const renderBreakdown = (title, breakdown, sectionKey) => {
    const entries = Object.entries(breakdown || {}).filter(([, value]) => Number.isFinite(value));
    const isOpen = openSections[sectionKey];
    return (
      <div style={cardStyle}>
        <button type="button" style={sectionHeaderStyle} onClick={() => toggleSection(sectionKey)}>
          <span>{title}</span>
          <span>{isOpen ? '▾' : '▸'}</span>
        </button>
        {isOpen && (
          entries.length === 0 ? (
            <div style={{ color: '#94a3b8', fontSize: '0.85rem', marginTop: '0.6rem' }}>No reward data yet.</div>
          ) : (
            <div style={{ marginTop: '0.6rem' }}>
              {entries.map(([key, value]) => (
                <div
                  key={key}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    gap: '0.75rem',
                    padding: '0.2rem 0',
                    borderBottom: '1px solid rgba(148, 163, 184, 0.1)',
                  }}
                >
                  <span style={{ color: '#cbd5e1', fontSize: '0.85rem' }}>{key}</span>
                  <span style={{ fontFamily: 'ui-monospace, SFMono-Regular, monospace', fontSize: '0.85rem' }}>
                    {Number(value).toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          )
        )}
      </div>
    );
  };

  return (
    <aside style={panelStyle}>
      <div
        onMouseDown={startDrag}
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '0.4rem',
          cursor: 'grab',
          paddingBottom: '0.35rem',
          borderBottom: '1px solid rgba(148, 163, 184, 0.18)',
        }}
      >
        <div>
          <div style={{ fontSize: '1.1rem', fontWeight: 800 }}>Reward Sidebar</div>
          <div style={{ color: '#93c5fd', fontSize: '0.9rem', marginTop: '0.25rem' }}>{envName}</div>
        </div>
        <div style={{ color: '#94a3b8', fontSize: '0.78rem' }}>{isDragging ? 'dragging' : 'drag me'}</div>
      </div>

      <div style={cardStyle}>
        <button type="button" style={sectionHeaderStyle} onClick={() => toggleSection('rewardTerms')}>
          <span>Reward Terms</span>
          <span>{openSections.rewardTerms ? '▾' : '▸'}</span>
        </button>
        {openSections.rewardTerms && (
          <>
            {!sidebarSupportsCustomReward && (
              <div style={{ color: '#94a3b8', fontSize: '0.85rem', marginTop: '0.75rem', marginBottom: '0.75rem' }}>
                This environment currently exposes only the native Gym reward. The shaping API is in place for more env-specific terms.
              </div>
            )}
            {!canTuneFromBreakdown && (
              <div style={{ color: '#94a3b8', fontSize: '0.85rem', marginTop: '0.75rem', marginBottom: '0.75rem' }}>
                Reward terms will appear here once a rollout publishes its breakdown.
              </div>
            )}
            {editableTerms.map((term) => {
              const isOpen = expandedTerms[term.key];
              return (
                <div
                  key={term.key}
                  style={{
                    borderTop: '1px solid rgba(148, 163, 184, 0.12)',
                    paddingTop: '0.6rem',
                    marginTop: '0.6rem',
                  }}
                >
                  <button
                    type="button"
                    onClick={() => toggleTerm(term.key)}
                    style={{
                      width: '100%',
                      background: 'transparent',
                      border: 'none',
                      color: '#e5eefb',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: 0,
                      cursor: 'pointer',
                      textAlign: 'left',
                    }}
                  >
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
                      <input
                        type="checkbox"
                        checked={term.enabled}
                        onChange={(event) => onTermChange(term.key, 'enabled', event.target.checked, term)}
                        onClick={(event) => event.stopPropagation()}
                      />
                      <span>{term.label}</span>
                    </span>
                    <span>{isOpen ? '▾' : '▸'}</span>
                  </button>
                  {isOpen && (
                    <>
                      <div style={{ color: '#94a3b8', fontSize: '0.78rem', marginTop: '0.35rem', lineHeight: 1.4 }}>
                        {term.description}
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.55rem' }}>
                        <span style={{ fontSize: '0.8rem', color: '#cbd5e1' }}>Weight</span>
                        <input
                          type="number"
                          step="0.01"
                          value={term.weight}
                          disabled={!term.enabled}
                          onChange={(event) => onTermChange(term.key, 'weight', Number(event.target.value), term)}
                          style={{
                            width: '96px',
                            padding: '0.35rem 0.45rem',
                            borderRadius: '8px',
                            border: '1px solid rgba(148, 163, 184, 0.3)',
                            backgroundColor: 'rgba(15, 23, 42, 0.65)',
                            color: '#f8fafc',
                          }}
                        />
                      </div>
                    </>
                  )}
                </div>
              );
            })}

            <button
              onClick={onSaveConfig}
              disabled={rewardConfigLoading || !rewardConfigDirty}
              style={{
                width: '100%',
                marginTop: '0.9rem',
                padding: '0.7rem 0.9rem',
                borderRadius: '10px',
                border: 'none',
                fontWeight: 700,
                cursor: rewardConfigLoading || !rewardConfigDirty ? 'not-allowed' : 'pointer',
                backgroundColor: rewardConfigDirty ? '#38bdf8' : '#334155',
                color: '#0f172a',
              }}
            >
              {rewardConfigLoading ? 'Saving...' : rewardConfigDirty ? 'Apply Reward Changes' : 'Reward Settings Applied'}
            </button>
            <div style={{ marginTop: '0.55rem', color: '#94a3b8', fontSize: '0.8rem' }}>{rewardConfigStatus}</div>
          </>
        )}
      </div>

      {renderBreakdown('Latest Training Breakdown', latestTrainingBreakdown, 'latestTraining')}
      {renderBreakdown('Training Mean Breakdown', latestTrainingMeanBreakdown, 'trainingMean')}
      {renderBreakdown('Latest Rollout Breakdown', latestRolloutBreakdown, 'latestRollout')}

      <div style={cardStyle}>
        <button type="button" style={sectionHeaderStyle} onClick={() => toggleSection('logs')}>
          <span>Recent Reward Logs</span>
          <span>{openSections.logs ? '▾' : '▸'}</span>
        </button>
        {openSections.logs && (
          safeRewardLogs.length === 0 ? (
            <div style={{ color: '#94a3b8', fontSize: '0.85rem', marginTop: '0.6rem' }}>No reward logs yet.</div>
          ) : (
            <div style={{ marginTop: '0.6rem' }}>
              {safeRewardLogs.slice(0, 10).map((entry, index) => (
                <div
                  key={`${entry.source}-${entry.label}-${index}`}
                  style={{
                    borderTop: index === 0 ? 'none' : '1px solid rgba(148, 163, 184, 0.12)',
                    paddingTop: index === 0 ? 0 : '0.55rem',
                    marginTop: index === 0 ? 0 : '0.55rem',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.75rem' }}>
                    <span style={{ fontWeight: 700 }}>{entry.label}</span>
                    <span style={{ color: '#7dd3fc' }}>{entry.source}</span>
                  </div>
                  <div style={{ color: '#cbd5e1', fontSize: '0.82rem', marginTop: '0.2rem' }}>
                    Total: {Number(entry.total || 0).toFixed(3)}
                  </div>
                  <div style={{ color: '#94a3b8', fontSize: '0.78rem', marginTop: '0.15rem' }}>{entry.at}</div>
                </div>
              ))}
            </div>
          )
        )}
      </div>
    </aside>
  );
}

export default RewardSidebar;
