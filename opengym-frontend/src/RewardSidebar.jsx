import { useEffect, useMemo, useRef, useState } from 'react';

function RewardSidebar({
  envName,
  rewardConfig,
  rewardConfigDirty,
  rewardConfigLoading,
  rewardConfigStatus,
  supportsCustomReward,
  availableRewardVariables,
  rewardFormulaExamples,
  rewardSourceLinks,
  latestTrainingBreakdown,
  latestTrainingMeanBreakdown,
  latestRolloutBreakdown,
  rewardLogs,
  onTermChange,
  onAddCustomTerm,
  onRemoveTerm,
  onSaveConfig,
}) {
  const safeRewardConfig = Array.isArray(rewardConfig) ? rewardConfig : [];
  const safeRewardLogs = Array.isArray(rewardLogs) ? rewardLogs : [];
  const safeRewardVariables = Array.isArray(availableRewardVariables) ? availableRewardVariables : [];
  const safeFormulaExamples = Array.isArray(rewardFormulaExamples) ? rewardFormulaExamples : [];
  const safeRewardSourceLinks = Array.isArray(rewardSourceLinks) ? rewardSourceLinks : [];
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
          expression: '',
          is_custom: false,
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
    formulaHelp: false,
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
          {safeRewardSourceLinks.length > 0 && (
            <div style={{ display: 'flex', gap: '0.45rem', flexWrap: 'wrap', marginTop: '0.45rem' }}>
              {safeRewardSourceLinks.map((link) => (
                <a
                  key={link.path}
                  href={link.path}
                  style={{
                    fontSize: '0.75rem',
                    color: '#bae6fd',
                    textDecoration: 'none',
                    padding: '0.2rem 0.5rem',
                    border: '1px solid rgba(186, 230, 253, 0.25)',
                    borderRadius: '999px',
                    backgroundColor: 'rgba(14, 116, 144, 0.18)',
                  }}
                >
                  {link.label}
                </a>
              ))}
            </div>
          )}
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
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      {term.is_custom && (
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            onRemoveTerm(term.key);
                          }}
                          style={{
                            border: '1px solid rgba(248, 113, 113, 0.35)',
                            backgroundColor: 'rgba(127, 29, 29, 0.4)',
                            color: '#fecaca',
                            borderRadius: '999px',
                            padding: '0.15rem 0.55rem',
                            fontSize: '0.72rem',
                            cursor: 'pointer',
                          }}
                        >
                          Remove
                        </button>
                      )}
                      <span>{isOpen ? '▾' : '▸'}</span>
                    </span>
                  </button>
                  {isOpen && (
                    <>
                      <div style={{ color: '#94a3b8', fontSize: '0.78rem', marginTop: '0.35rem', lineHeight: 1.4 }}>
                        {term.description}
                      </div>
                      {term.is_custom && (
                        <>
                          <div style={{ marginTop: '0.55rem' }}>
                            <div style={{ color: '#cbd5e1', fontSize: '0.8rem', marginBottom: '0.25rem' }}>Key</div>
                            <input
                              type="text"
                              value={term.key}
                              onChange={(event) => onTermChange(term.key, 'key', event.target.value, term)}
                              style={{
                                width: '100%',
                                padding: '0.45rem 0.55rem',
                                borderRadius: '8px',
                                border: '1px solid rgba(148, 163, 184, 0.3)',
                                backgroundColor: 'rgba(15, 23, 42, 0.65)',
                                color: '#f8fafc',
                              }}
                            />
                          </div>
                          <div style={{ marginTop: '0.55rem' }}>
                            <div style={{ color: '#cbd5e1', fontSize: '0.8rem', marginBottom: '0.25rem' }}>Label</div>
                            <input
                              type="text"
                              value={term.label}
                              onChange={(event) => onTermChange(term.key, 'label', event.target.value, term)}
                              style={{
                                width: '100%',
                                padding: '0.45rem 0.55rem',
                                borderRadius: '8px',
                                border: '1px solid rgba(148, 163, 184, 0.3)',
                                backgroundColor: 'rgba(15, 23, 42, 0.65)',
                                color: '#f8fafc',
                              }}
                            />
                          </div>
                        </>
                      )}
                      <div style={{ marginTop: '0.55rem' }}>
                        <div style={{ color: '#cbd5e1', fontSize: '0.8rem', marginBottom: '0.25rem' }}>Expression</div>
                        <textarea
                          value={term.expression || ''}
                          onChange={(event) => onTermChange(term.key, 'expression', event.target.value, term)}
                          rows={3}
                          style={{
                            width: '100%',
                            padding: '0.5rem 0.6rem',
                            borderRadius: '8px',
                            border: '1px solid rgba(148, 163, 184, 0.3)',
                            backgroundColor: 'rgba(15, 23, 42, 0.65)',
                            color: '#f8fafc',
                            resize: 'vertical',
                            fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                            fontSize: '0.8rem',
                          }}
                        />
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
              type="button"
              onClick={onAddCustomTerm}
              style={{
                width: '100%',
                marginTop: '0.8rem',
                padding: '0.65rem 0.9rem',
                borderRadius: '10px',
                border: '1px dashed rgba(125, 211, 252, 0.45)',
                backgroundColor: 'rgba(14, 116, 144, 0.18)',
                color: '#bae6fd',
                fontWeight: 700,
                cursor: 'pointer',
              }}
            >
              + Add Custom Reward Term
            </button>

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

            <div style={{ marginTop: '0.8rem', borderTop: '1px solid rgba(148, 163, 184, 0.12)', paddingTop: '0.8rem' }}>
              <button type="button" style={sectionHeaderStyle} onClick={() => toggleSection('formulaHelp')}>
                <span>Formula Variables</span>
                <span>{openSections.formulaHelp ? '▾' : '▸'}</span>
              </button>
              {openSections.formulaHelp && (
                <>
                  <div style={{ color: '#94a3b8', fontSize: '0.78rem', marginTop: '0.5rem', lineHeight: 1.45 }}>
                    Use these variable names inside custom reward expressions. The backend supports plain arithmetic plus helper functions like abs, min, max, clip, sqrt, square, exp, log, sin, cos, tanh, and sign.
                  </div>
                  {safeFormulaExamples.length > 0 && (
                    <div style={{ marginTop: '0.55rem' }}>
                      {safeFormulaExamples.map((example) => (
                        <div
                          key={example}
                          style={{
                            fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                            fontSize: '0.75rem',
                            color: '#cbd5e1',
                            padding: '0.22rem 0',
                          }}
                        >
                          {example}
                        </div>
                      ))}
                    </div>
                  )}
                  <div style={{ marginTop: '0.55rem', maxHeight: '180px', overflowY: 'auto' }}>
                    {safeRewardVariables.map((variable) => (
                      <div
                        key={variable.name}
                        style={{
                          borderTop: '1px solid rgba(148, 163, 184, 0.08)',
                          padding: '0.35rem 0',
                          textAlign: 'left',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.45rem', flexWrap: 'wrap' }}>
                          <div style={{ fontSize: '0.78rem', color: '#e2e8f0', fontFamily: 'ui-monospace, SFMono-Regular, monospace' }}>
                            {variable.name}
                          </div>
                          {variable.display_name && variable.display_name !== variable.name && (
                            <div style={{ fontSize: '0.72rem', color: '#7dd3fc', fontFamily: 'ui-monospace, SFMono-Regular, monospace' }}>
                              {variable.display_name}
                            </div>
                          )}
                        </div>
                        <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>
                          {variable.description}
                        </div>
                        {Array.isArray(variable.aliases) && variable.aliases.length > 0 && (
                          <div style={{ fontSize: '0.68rem', color: '#64748b', marginTop: '0.16rem' }}>
                            Aliases: {variable.aliases.join(', ')}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
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
