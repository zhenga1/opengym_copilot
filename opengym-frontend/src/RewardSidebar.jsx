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
  const panelStyle = {
    width: '340px',
    minWidth: '300px',
    background: 'linear-gradient(180deg, #0f172a, #111827)',
    color: '#e5eefb',
    borderRadius: '16px',
    padding: '1rem',
    boxShadow: '0 12px 28px rgba(15, 23, 42, 0.28)',
    position: 'sticky',
    top: '1rem',
  };

  const cardStyle = {
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
    border: '1px solid rgba(148, 163, 184, 0.18)',
    borderRadius: '12px',
    padding: '0.85rem',
    marginTop: '0.9rem',
  };

  const renderBreakdown = (title, breakdown) => {
    const entries = Object.entries(breakdown || {}).filter(([, value]) => Number.isFinite(value));
    return (
      <div style={cardStyle}>
        <div style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: '0.5rem' }}>{title}</div>
        {entries.length === 0 ? (
          <div style={{ color: '#94a3b8', fontSize: '0.85rem' }}>No reward data yet.</div>
        ) : (
          entries.map(([key, value]) => (
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
          ))
        )}
      </div>
    );
  };

  return (
    <aside style={panelStyle}>
      <div style={{ fontSize: '1.1rem', fontWeight: 800 }}>Reward Sidebar</div>
      <div style={{ color: '#93c5fd', fontSize: '0.9rem', marginTop: '0.25rem' }}>{envName}</div>

      <div style={cardStyle}>
        <div style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: '0.6rem' }}>Reward Terms</div>
        {!supportsCustomReward && (
          <div style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '0.75rem' }}>
            This environment currently exposes only the native Gym reward. The shaping API is in place for more env-specific terms.
          </div>
        )}
        {rewardConfig.map((term) => (
          <div
            key={term.key}
            style={{
              borderTop: '1px solid rgba(148, 163, 184, 0.12)',
              paddingTop: '0.6rem',
              marginTop: '0.6rem',
            }}
          >
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
              <input
                type="checkbox"
                checked={term.enabled}
                onChange={(event) => onTermChange(term.key, 'enabled', event.target.checked)}
              />
              <span>{term.label}</span>
            </label>
            <div style={{ color: '#94a3b8', fontSize: '0.78rem', marginTop: '0.2rem', lineHeight: 1.4 }}>
              {term.description}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.45rem' }}>
              <span style={{ fontSize: '0.8rem', color: '#cbd5e1' }}>Weight</span>
              <input
                type="number"
                step="0.01"
                value={term.weight}
                disabled={!term.enabled}
                onChange={(event) => onTermChange(term.key, 'weight', Number(event.target.value))}
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
          </div>
        ))}

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
      </div>

      {renderBreakdown('Latest Training Breakdown', latestTrainingBreakdown)}
      {renderBreakdown('Training Mean Breakdown', latestTrainingMeanBreakdown)}
      {renderBreakdown('Latest Rollout Breakdown', latestRolloutBreakdown)}

      <div style={cardStyle}>
        <div style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: '0.6rem' }}>Recent Reward Logs</div>
        {rewardLogs.length === 0 ? (
          <div style={{ color: '#94a3b8', fontSize: '0.85rem' }}>No reward logs yet.</div>
        ) : (
          rewardLogs.slice(0, 10).map((entry, index) => (
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
          ))
        )}
      </div>
    </aside>
  );
}

export default RewardSidebar;
