import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { createPortal } from 'react-dom'
import {Line} from 'react-chartjs-2'
import SetPathPopup from './SetPathPopup'
import SaveRolloutPopup from './RolloutPopup'
import ProgressBar from './ProgressBar'
import { isCancel } from 'axios'
import apiClient from './apiClient'
import {Chart as ChartJS, LineElement, CategoryScale, LinearScale, PointElement} from 'chart.js'
import { buildWebSocketUrl } from './runtimeConfig'

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement);

function InsightsModal({ isOpen, onClose, title, accentColor, description, children }) {
  useEffect(() => {
    if (!isOpen) {
      return undefined;
    }
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [isOpen]);

  if (!isOpen) {
    return null;
  }

  // This renders the Insight Modal component as a portal
  // What does this mean?
  // We see that it renders the children within the popup-card--insights div and then 
  // puts it within the div of the popup-backdrop. However, instead of rendering the whole structure inside of popup-backdrop,
  // it renders the structure inside of popup-card--insights. This allows the insight modal to be rendered outside of the normal React component hierarchy, 
  // which can be useful for modals and popups that need to overlay other content on the page without being affected by the parent components' styles or layout. 
  // By using createPortal, we can ensure that the modal is rendered at the top level of the DOM, allowing it to function properly as an overlay.
  
  // This createPortal renders the InsightsModel as a child of the document.body element, ouside of hierarchy
  // of react. 
  return createPortal(
    <div className="popup-backdrop" onClick={onClose}>
      <div className="popup-card popup-card--insights" onClick={(event) => event.stopPropagation()}>
        <div className="popup-header">
          <div>{title}</div>
          <button className="popup-close" onClick={onClose} aria-label="Close">x</button>
        </div>
        <div className="popup-body popup-body--scroll">
          <div style={{ color: accentColor, fontSize: '1.1rem', fontWeight: 800, marginBottom: '0.45rem' }}>
            {title}
          </div>
          <div style={{ color: '#64748b', fontSize: '0.9rem', marginBottom: '0.95rem', lineHeight: 1.55 }}>
            {description}
          </div>
          {children}
        </div>
        <div className="popup-actions">
          <button className="btn secondary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>,
    document.body
  );
}

function RolloutWindow({
  isActive = false,
  onSidebarStateChange = () => {},
  // special argument called by the parent component 
  onOpenLoadedRollout = () => {},
  viewerMode = 'live',
  initialRollouts = [],
  initialEnvName = 'CartPole-v1',
  viewerLabel = '',
}) {
  const isSavedViewer = viewerMode === 'saved';
  const [rollouts, setRollouts] = useState([]);
  const [trainingRollouts, setTrainingRollouts] = useState([]);
  const [trainingEpisodes, setTrainingEpisodes] = useState([]);
  const [frames, setFrames] = useState([]);
  const [capturedEpisodeFramesByEpisode, setCapturedEpisodeFramesByEpisode] = useState({});
  const [selectedVisualizationEpisode, setSelectedVisualizationEpisode] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [stepInterval, setStepInterval] = useState(5);
  const [hoveronReloadTempModels, setHoveronReloadTempModels] = useState(false); // [false,]
  // defined
  const [renewFrameInterval, setRenewFrameInterval] = useState(5);
  const [replayInterval, setReplayInterval] = useState(50); // in ms
  const [episodeNumForSimulation, setEpisodeNumForSimulation] = useState(0);
  // whether to train in the backend
  const [trainSteps, setTrainSteps] = useState(1000);
  const [trainMode, setTrainMode] = useState(false);
  const [stoppingTraining, setStoppingTraining] = useState(false);
  // Stores info on the CURRENT episode
  const [episodeInfo, setEpisodeInfo] = useState({episode: 0, reward: 0});
  const [rewardConfig, setRewardConfig] = useState([]);
  const [rewardConfigDirty, setRewardConfigDirty] = useState(false);
  const [rewardConfigLoading, setRewardConfigLoading] = useState(false);
  const [rewardConfigStatus, setRewardConfigStatus] = useState("Loading reward terms...");
  const [supportsCustomReward, setSupportsCustomReward] = useState(false);
  const [availableRewardVariables, setAvailableRewardVariables] = useState([]);
  const [rewardFormulaExamples, setRewardFormulaExamples] = useState([]);
  const [rewardSourceLinks, setRewardSourceLinks] = useState([]);
  const [savedRewardConfigFiles, setSavedRewardConfigFiles] = useState([]);
  const [selectedRewardConfigFile, setSelectedRewardConfigFile] = useState('');
  const [rewardConfigSaveName, setRewardConfigSaveName] = useState('');
  const [rewardConfigSaveSourceType, setRewardConfigSaveSourceType] = useState('manual');
  const [taskGoal, setTaskGoal] = useState('');
  const [taskProposal, setTaskProposal] = useState(null);
  const [taskProposalLoading, setTaskProposalLoading] = useState(false);
  const [taskProposalStatus, setTaskProposalStatus] = useState('Describe a task goal, then generate a proposal.');
  const [taskProposalLiveStatus, setTaskProposalLiveStatus] = useState(null);
  const [availableBehaviorTags, setAvailableBehaviorTags] = useState([]);
  const [availableLlms, setAvailableLlms] = useState([]);
  const [selectedLlmId, setSelectedLlmId] = useState('');
  // "fallback" = heuristic on first failure; "retry" = re-query LLM until a proposal validates
  const [proposalStrategy, setProposalStrategy] = useState('fallback');
  const [trainingRewardBreakdown, setTrainingRewardBreakdown] = useState({});
  const [trainingRewardBreakdownMean, setTrainingRewardBreakdownMean] = useState({});
  const [trainingAblationReport, setTrainingAblationReport] = useState(null);
  const [trainingAblationStatus, setTrainingAblationStatus] = useState('idle');
  const [trainingInsights, setTrainingInsights] = useState(null);
  const [trainingBehaviorReport, setTrainingBehaviorReport] = useState(null);
  const [trainingBehaviorTags, setTrainingBehaviorTags] = useState(null);
  const [showTrainingInsights, setShowTrainingInsights] = useState(false);
  const [trainingGraphIndex, setTrainingGraphIndex] = useState(0);
  const [trainingWorkspaceViewIndex, setTrainingWorkspaceViewIndex] = useState(0);
  const [trainingTimelineGraphIndex, setTrainingTimelineGraphIndex] = useState(0);
  const [selectedTrainingTimelineEpisode, setSelectedTrainingTimelineEpisode] = useState(null);
  const [trainingTimelineMode, setTrainingTimelineMode] = useState('episode');
  const [trainingTimelineOutcomeFilter, setTrainingTimelineOutcomeFilter] = useState('all');
  const [rolloutGraphIndex, setRolloutGraphIndex] = useState(0);
  const [timelineGraphIndex, setTimelineGraphIndex] = useState(0);
  const [selectedTimelineEpisode, setSelectedTimelineEpisode] = useState(null);
  const [selectedTimelineStep, setSelectedTimelineStep] = useState(null);
  const [rolloutTimelineMode, setRolloutTimelineMode] = useState('episode');
  const [rolloutTimelineOutcomeFilter, setRolloutTimelineOutcomeFilter] = useState('all');
  const [rolloutWorkspaceViewIndex, setRolloutWorkspaceViewIndex] = useState(0);
  const [showRolloutInsights, setShowRolloutInsights] = useState(false);
  const [rolloutRewardBreakdown, setRolloutRewardBreakdown] = useState({});
  const [rolloutInsights, setRolloutInsights] = useState(null);
  const [rolloutBehaviorReport, setRolloutBehaviorReport] = useState(null);
  const [rolloutBehaviorTags, setRolloutBehaviorTags] = useState(null);
  const [latestRolloutRawTerms, setLatestRolloutRawTerms] = useState({});
  const [rewardLogs, setRewardLogs] = useState([]);

  const [envName, setEnvName] = useState(initialEnvName);
  const [isPaused, setIsPaused] = useState(false);

  // the FPS of rollout, default is 20FPS (delay = 1/20 = 0.05 seconds)
  const [rolloutSpeed, setRolloutSpeed] = useState(20);
  const [showSavePopup, setShowSavePopup] = useState(false);

  // Upload the files logistics:
  const [serverModels, setServerModels] = useState([]);
  const [serverModelRecords, setServerModelRecords] = useState([]);
  const [selectedServerModel, setSelectedServerModel] = useState(""); // "" = None
  const [activeModelName, setActiveModelName] = useState("");
  const [modelSortOrder, setModelSortOrder] = useState('newest');
  const [rolloutFiles, setRolloutFiles] = useState([]);
  const [selectedRolloutFile, setSelectedRolloutFile] = useState("");
  const [showPopup, setShowPopup] = useState(false);
  const [file, setFile] = useState(null);
  // whether is using default policy or not
  const [isUsingNone, setIsUsingNone] = useState(true);
  const [loading, setLoading] = useState(false);
  const [loadingSavedRollouts, setLoadingSavedRollouts] = useState(false);
  const sortedServerModelRecords = useMemo(() => {
    const records = Array.isArray(serverModelRecords) ? [...serverModelRecords] : [];
    records.sort((left, right) => {
      const leftTs = Number(left?.created_ts || 0);
      const rightTs = Number(right?.created_ts || 0);
      if (modelSortOrder === 'oldest') {
        return leftTs - rightTs || String(left?.name || '').localeCompare(String(right?.name || ''));
      }
      return rightTs - leftTs || String(left?.name || '').localeCompare(String(right?.name || ''));
    });
    return records;
  }, [modelSortOrder, serverModelRecords]);



  const selectedServerModelRecord = useMemo(
    () => sortedServerModelRecords.find((record) => record.name === selectedServerModel) || null,
    [selectedServerModel, sortedServerModelRecords]
  );
  const activeModelRecord = useMemo(
    () => sortedServerModelRecords.find((record) => record.name === activeModelName) || null,
    [activeModelName, sortedServerModelRecords]
  );
  //set whether model parent directory file path is copied
  const [filePathCopied, setFilePathCopied] = useState(false);
  const [hoverOnFilePathButton, setHoverOnFilePathButton] = useState(false);
  const [hoverOnDeleteAllTemp, setHoverOnDeleteAllTempButton] = useState(false);

  const setShowSavePopupToTrue = () => {
    setSavePopupMode('manual');
    setPendingModelSwitch(null);
    setShowSavePopup(true);
  };
  const closeShowSavePopup = () => {
    setSavePopupMode('manual');
    setPendingModelSwitch(null);
    setShowSavePopup(false);
  };
  const [savePopupMode, setSavePopupMode] = useState('manual');
  const [pendingModelSwitch, setPendingModelSwitch] = useState(null);

  const isPausedRef = useRef(false);
  const [sessionId, setSessionId] = useState(null);
  // whether or not the current rollout (all the rewards) is being saved
  const [saving_rollouts, setSavingRollouts] = useState(false);
  const [runId, setRunId] = useState(null);
  const [rolloutSessionVersion, setRolloutSessionVersion] = useState(0);
  const intervalRef = useRef(null);
  const rolloutEpisodeOffsetRef = useRef(null);

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
  const rewardLogLimit = 30;
  const terminalHighlightWindow = 8;
  const rolloutChartMinWidth = 600;
  const rolloutChartPointWidth = 36;
  const rolloutChartBucketSize = 25;

  const appendRewardLog = useCallback((entry) => {
    setRewardLogs((prev) => [entry, ...prev.slice(0, rewardLogLimit - 1)]);
  }, []);

  const formatModelTimestamp = useCallback((value) => {
    if (!value) return 'Unknown date';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return 'Unknown date';
    return date.toLocaleString();
  }, []);

  const formatModelSize = useCallback((sizeBytes) => {
    const size = Number(sizeBytes || 0);
    if (!Number.isFinite(size) || size <= 0) return 'Unknown size';
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
    return `${(size / (1024 * 1024)).toFixed(2)} MB`;
  }, []);

  const handleEnvChange = (e) => {
    if (isSavedViewer) return;
    setEnvName(e.target.value)
  }

  // This is for the load rollouts section
  const hydrateLoadedRollouts = useCallback((loadedRollouts) => {
    const safeRollouts = Array.isArray(loadedRollouts) ? loadedRollouts : [];
    const earliestEpisode = safeRollouts.reduce((minEpisode, rollout) => {
      const episode = Number(rollout?.episode);
      if (!Number.isFinite(episode)) return minEpisode;
      if (minEpisode === null || episode < minEpisode) return episode;
      return minEpisode;
    }, null);
    const initialTimelineRollout =
      earliestEpisode === null
        ? safeRollouts[0] || {}
        : safeRollouts.find((rollout) => Number(rollout?.episode) === earliestEpisode) || safeRollouts[0] || {};
    setRollouts(safeRollouts);
    setTrainingRollouts([]);
    setTrainingEpisodes([]);
    setFrames([]);
    setCapturedEpisodeFramesByEpisode({});
    setSelectedVisualizationEpisode(null);
    setCurrentFrame(0);
    setIsPlaying(false);
    setTrainingRewardBreakdown({});
    setTrainingRewardBreakdownMean({});

    const latest = safeRollouts[0] || {};
    setEpisodeInfo({
      episode: latest.episode ?? 0,
      reward: latest.reward ?? 0,
    });
    setEpisodeNumForSimulation(latest.episode ?? 0);
    setRolloutRewardBreakdown(latest.reward_breakdown || {});
    setLatestRolloutRawTerms(latest.reward_raw_terms || {});
    setSelectedTimelineEpisode(initialTimelineRollout.episode ?? null);
    setSelectedTimelineStep(
      initialTimelineRollout.episode_terminal_timestep ??
      (Array.isArray(initialTimelineRollout.reward_history) ? initialTimelineRollout.reward_history.length : null)
    );
    setRewardLogs(
      safeRollouts.slice(0, rewardLogLimit).map((rollout) => ({
        source: 'loaded',
        label: `Episode ${rollout.episode ?? 0}`,
        total: rollout.reward ?? 0,
        breakdown: rollout.reward_breakdown || {},
        at: viewerLabel || 'saved rollout',
      }))
    );
  }, [rewardLogLimit, viewerLabel]);

  const clearLiveRolloutState = useCallback(() => {
    // Clear the live rollout state and reset the rollout UI
    rolloutEpisodeOffsetRef.current = null;
    setRollouts([]);
    setFrames([]);
    setCapturedEpisodeFramesByEpisode({});
    setSelectedVisualizationEpisode(null);
    setCurrentFrame(0);
    setIsPlaying(false);
    setEpisodeInfo({ episode: 0, reward: 0 });
    setEpisodeNumForSimulation(0);
    setRolloutRewardBreakdown({});
    setLatestRolloutRawTerms({});
    setSelectedTimelineEpisode(null);
    setSelectedTimelineStep(null);
    setRewardLogs([]);
    setRolloutInsights(null);
    setRolloutBehaviorReport(null);
    setRolloutBehaviorTags(null);
  }, []);

  const restartLiveRolloutSession = useCallback(() => {
    clearLiveRolloutState();
    setSessionId(null);
    setRolloutSessionVersion((prev) => prev + 1);
  }, [clearLiveRolloutState]);

  const updateRewardTerm = useCallback((termKey, field, value, fallbackTerm = null) => {
    setRewardConfig((prev) => {
      const nextValue = field === 'weight' ? Number(value) : value;
      const existingIndex = prev.findIndex((term) => term.key === termKey);

      if (existingIndex >= 0) {
        return prev.map((term) =>
          term.key === termKey ? { ...term, [field]: nextValue } : term
        );
      }

      const seededTerm = fallbackTerm ?? {
        key: termKey,
        label: termKey.replaceAll('_', ' ').replace(/\b\w/g, (char) => char.toUpperCase()),
        description: 'Recovered from rollout reward breakdown.',
        enabled: true,
        weight: 1,
        expression: '',
        is_custom: false,
      };

      return [...prev, { ...seededTerm, [field]: nextValue }];
    });
    setRewardConfigDirty(true);
    setRewardConfigStatus("Unsaved reward changes.");
  }, []);

  const addCustomRewardTerm = useCallback(() => {
    setRewardConfig((prev) => {
      let index = prev.filter((term) => term.is_custom).length + 1;
      let key = `custom_term_${index}`;
      while (prev.some((term) => term.key === key)) {
        index += 1;
        key = `custom_term_${index}`;
      }
      return [
        ...prev,
        {
          key,
          label: `Custom Term ${index}`,
          description: 'User-defined reward term computed from the formula below.',
          enabled: true,
          weight: 1.0,
          expression: '0',
          is_custom: true,
        },
      ];
    });
    setRewardConfigDirty(true);
    setRewardConfigStatus('Added custom reward term. Define its formula, then apply reward changes.');
  }, []);

  const removeRewardTerm = useCallback((termKey) => {
    setRewardConfig((prev) => prev.filter((term) => term.key !== termKey));
    setRewardConfigDirty(true);
    setRewardConfigStatus('Removed custom reward term. Apply reward changes to persist.');
  }, []);

  const showSavedViewerRewardMessage = useCallback(() => {
    setRewardConfigStatus('Saved rollout viewers are read-only. Edit reward terms from a live rollout window.');
  }, []);

  const showSavedViewerTaskMessage = useCallback(() => {
    setTaskProposalStatus('Saved rollout viewers are read-only. Generate and apply task configs from a live rollout window.');
  }, []);

  const normalizeTaskProposal = useCallback((proposal) => {
    if (!proposal || typeof proposal !== 'object') return null;
    const normalizedTaskParams = Array.isArray(proposal.task_params)
      ? proposal.task_params.map((param) => ({
          key: String(param?.key || ''),
          value: param?.value,
          description:
            typeof param?.description === 'string'
              ? param.description
              : typeof param?.description?.description === 'string'
                ? param.description.description
                : typeof param?.description?.expression === 'string'
                  ? param.description.expression
                  : '',
        }))
      : [];
    const normalizedDerivedSignals = Array.isArray(proposal.derived_signals)
      ? proposal.derived_signals.map((signal) => ({
          key: String(signal?.key || ''),
          expression:
            typeof signal?.expression === 'string'
              ? signal.expression
              : typeof signal?.expression?.expression === 'string'
                ? signal.expression.expression
                : '',
          description:
            typeof signal?.description === 'string'
              ? signal.description
              : typeof signal?.description?.description === 'string'
                ? signal.description.description
                : typeof signal?.description?.expression === 'string'
                  ? signal.description.expression
                  : '',
        }))
      : [];
    const normalizedRewardTerms = Array.isArray(proposal.reward_terms)
      ? proposal.reward_terms.map((term) => ({
          ...term,
          key: String(term?.key || ''),
          label: String(term?.label || term?.key || ''),
          weight: Number.isFinite(Number(term?.weight)) ? Number(term.weight) : 0,
          enabled: typeof term?.enabled === 'boolean' ? term.enabled : true,
          description:
            typeof term?.description === 'string'
              ? term.description
              : typeof term?.description?.description === 'string'
                ? term.description.description
                : typeof term?.description?.expression === 'string'
                  ? term.description.expression
                  : '',
          expression:
            typeof term?.expression === 'string'
              ? term.expression
              : typeof term?.expression?.expression === 'string'
                ? term.expression.expression
                : '',
        }))
      : [];
    const normalizeBehaviorPlanItems = (items) =>
      Array.isArray(items)
        ? items.map((item) => ({
            key: String(item?.key || ''),
            weight: Number(item?.weight ?? 0),
            reason: typeof item?.reason === 'string' ? item.reason : '',
          }))
        : [];
    const normalizedBehaviorPlan =
      proposal.behavior_plan && typeof proposal.behavior_plan === 'object'
        ? {
            ...proposal.behavior_plan,
            goal: typeof proposal.behavior_plan.goal === 'string' ? proposal.behavior_plan.goal : '',
            desired_tags: normalizeBehaviorPlanItems(proposal.behavior_plan.desired_tags),
            avoid_tags: normalizeBehaviorPlanItems(proposal.behavior_plan.avoid_tags),
            constraints: Array.isArray(proposal.behavior_plan.constraints)
              ? proposal.behavior_plan.constraints.map((item) => String(item))
              : [],
            rationale: typeof proposal.behavior_plan.rationale === 'string' ? proposal.behavior_plan.rationale : '',
          }
        : null;
    const normalizedAvailableBehaviorTags = Array.isArray(proposal.available_behavior_tags)
      ? proposal.available_behavior_tags.map((tag) => ({
          key: String(tag?.key || ''),
          title: String(tag?.title || tag?.key || ''),
          description: typeof tag?.description === 'string' ? tag.description : '',
          polarity: String(tag?.polarity || ''),
          tags: Array.isArray(tag?.tags) ? tag.tags.map((item) => String(item)) : [],
        }))
      : [];
    return {
      ...proposal,
      goal: typeof proposal.goal === 'string' ? proposal.goal : '',
      raw_model_response: typeof proposal.raw_model_response === 'string'
        ? proposal.raw_model_response
        : typeof proposal._raw_model_response === 'string'
          ? proposal._raw_model_response
          : '',
      llm_error_stage: typeof proposal.llm_error_stage === 'string' ? proposal.llm_error_stage : '',
      parse_recovered: Boolean(proposal._parse_recovered),
      model_proposal_preview:
        proposal.model_proposal_preview && typeof proposal.model_proposal_preview === 'object'
          ? proposal.model_proposal_preview
          : null,
      success_metric:
        typeof proposal.success_metric === 'string'
          ? proposal.success_metric
          : typeof proposal.success_metric?.description === 'string'
            ? proposal.success_metric.description
            : typeof proposal.success_metric?.expression === 'string'
              ? proposal.success_metric.expression
              : '',
      rationale:
        typeof proposal.rationale === 'string'
          ? proposal.rationale
          : typeof proposal.rationale?.description === 'string'
            ? proposal.rationale.description
            : typeof proposal.rationale?.expression === 'string'
              ? proposal.rationale.expression
              : '',
      warnings: Array.isArray(proposal.warnings)
        ? proposal.warnings.map((warning) => String(warning))
        : [],
      behavior_plan: normalizedBehaviorPlan,
      available_behavior_tags: normalizedAvailableBehaviorTags,
      task_params: normalizedTaskParams,
      derived_signals: normalizedDerivedSignals,
      reward_terms: normalizedRewardTerms,
    };
  }, []);

  const buildProposalRewardPreviewTerms = useCallback((proposal) => {
    const terms = Array.isArray(proposal?.reward_terms) ? proposal.reward_terms : [];
    return terms.map((term, index) => ({
      key: String(term?.key || `proposal_term_${index}`),
      label: String(term?.label || term?.key || `Proposal Term ${index + 1}`),
      description:
        typeof term?.description === 'string'
          ? term.description
          : 'Proposed by the task-config planner. Review and apply to persist it on the backend.',
      weight: Number.isFinite(Number(term?.weight)) ? Number(term.weight) : 0,
      enabled: typeof term?.enabled === 'boolean' ? term.enabled : true,
      expression:
        typeof term?.expression === 'string'
          ? term.expression
          : '',
      is_custom: typeof term?.is_custom === 'boolean' ? term.is_custom : String(term?.key || '') !== 'native',
    }));
  }, []);

  const formatRewardTermLabel = useCallback((label) => {
    if (!label) return 'Reward Term';
    if (label === 'total_reward') return 'Total Reward';
    if (label.startsWith('obs_')) return `Observation ${label.slice(4)}`;
    if (label.startsWith('action_')) return `Action ${label.slice(7)}`;
    if (label.startsWith('prev_action_')) return `Previous Action ${label.slice(12)}`;
    return label
      .replaceAll('_', ' ')
      .replace(/\b\w/g, (char) => char.toUpperCase());
  }, []);

  const computeBreakdownFromRawTerms = useCallback((terms, rawTerms) => {
    const raw = rawTerms && Object.keys(rawTerms).length > 0 ? rawTerms : null;
    if (!raw) return null;

    const breakdown = {};
    let total = 0;
    for (const term of terms) {
      const rawValue = Number(raw[term.key] ?? 0);
      const contribution = term.enabled ? Number(term.weight) * rawValue : 0;
      breakdown[term.key] = contribution;
      total += contribution;
    }
    return { total, ...breakdown };
  }, []);

  const applyRewardConfigToRollouts = useCallback((entries, terms) => {
    return entries.map((entry) => {
      const nextRewardHistory = Array.isArray(entry.reward_history)
        ? entry.reward_history.map((stepEntry) => {
            const nextStepBreakdown = computeBreakdownFromRawTerms(terms, stepEntry.reward_raw_terms || {});
            if (!nextStepBreakdown) {
              return stepEntry;
            }
            return {
              ...stepEntry,
              reward: nextStepBreakdown.total,
              reward_breakdown: nextStepBreakdown,
            };
          })
        : [];
      const nextBreakdown = computeBreakdownFromRawTerms(terms, entry.reward_raw_terms || {});
      if (!nextBreakdown) {
        return {
          ...entry,
          reward_history: nextRewardHistory.length > 0 ? nextRewardHistory : entry.reward_history,
        };
      }
      return {
        ...entry,
        reward: nextBreakdown.total,
        reward_breakdown: nextBreakdown,
        reward_history: nextRewardHistory,
      };
    });
  }, [computeBreakdownFromRawTerms]);

  const applyRewardConfigToTrainingEpisodes = useCallback((entries, terms) => {
    return entries.map((entry) => {
      const nextRewardHistory = Array.isArray(entry.reward_history)
        ? entry.reward_history.map((stepEntry) => {
            const nextStepBreakdown = computeBreakdownFromRawTerms(terms, stepEntry.reward_raw_terms || {});
            if (!nextStepBreakdown) {
              return stepEntry;
            }
            return {
              ...stepEntry,
              reward: nextStepBreakdown.total,
              reward_breakdown: nextStepBreakdown,
            };
          })
        : [];
      const aggregatedBreakdown = nextRewardHistory.reduce((acc, stepEntry) => {
        const stepBreakdown = stepEntry.reward_breakdown || {};
        for (const [key, value] of Object.entries(stepBreakdown)) {
          acc[key] = (acc[key] || 0) + Number(value || 0);
        }
        return acc;
      }, {});
      const totalReward = nextRewardHistory.reduce((sum, stepEntry) => sum + Number(stepEntry.reward ?? 0), 0);
      return {
        ...entry,
        reward: nextRewardHistory.length > 0 ? totalReward : entry.reward,
        reward_breakdown: nextRewardHistory.length > 0 ? { ...aggregatedBreakdown, total: totalReward } : entry.reward_breakdown,
        reward_history: nextRewardHistory.length > 0 ? nextRewardHistory : entry.reward_history,
      };
    });
  }, [computeBreakdownFromRawTerms]);

  const summarizeEpisodeOutcome = useCallback((entry) => {
    if (!entry) {
      return {
        outcome: 'unknown',
        label: 'Outcome unavailable',
        detail: 'No episode selected.',
        terminalTimestep: null,
        highlightStart: null,
        highlightEnd: null,
        accentColor: 'rgba(100, 116, 139, 0.9)',
        shadeColor: 'rgba(148, 163, 184, 0.14)',
      };
    }

    const rewardHistory = Array.isArray(entry.reward_history) ? entry.reward_history : [];
    const inferredOutcome =
      entry.episode_outcome ||
      (entry.truncated && !entry.terminated ? 'success' : entry.terminated ? 'failure' : 'unknown');
    const terminalTimestep =
      Number(entry.episode_terminal_timestep) ||
      (rewardHistory.length > 0 ? rewardHistory.length : null);
    const highlightEnd = terminalTimestep;
    const highlightStart = terminalTimestep
      ? Math.max(1, terminalTimestep - terminalHighlightWindow + 1)
      : null;

    if (inferredOutcome === 'success') {
      return {
        outcome: 'success',
        label: terminalTimestep ? `Success at timestep ${terminalTimestep}` : 'Successful episode',
        detail: entry.episode_outcome_reason || 'Episode ended successfully.',
        terminalTimestep,
        highlightStart,
        highlightEnd,
        accentColor: 'rgba(22, 163, 74, 0.95)',
        shadeColor: 'rgba(34, 197, 94, 0.12)',
      };
    }

    if (inferredOutcome === 'failure') {
      return {
        outcome: 'failure',
        label: terminalTimestep ? `Failure at timestep ${terminalTimestep}` : 'Failed episode',
        detail: entry.episode_outcome_reason || 'Episode terminated in failure.',
        terminalTimestep,
        highlightStart,
        highlightEnd,
        accentColor: 'rgba(220, 38, 38, 0.95)',
        shadeColor: 'rgba(239, 68, 68, 0.12)',
      };
    }

    return {
      outcome: 'unknown',
      label: terminalTimestep ? `Terminal event at timestep ${terminalTimestep}` : 'Outcome unavailable',
      detail: entry.episode_outcome_reason || 'The environment did not report a clear success/failure signal.',
      terminalTimestep,
      highlightStart,
      highlightEnd,
      accentColor: 'rgba(100, 116, 139, 0.9)',
      shadeColor: 'rgba(148, 163, 184, 0.14)',
    };
  }, []);

  const buildTerminalHighlightPlugin = useCallback((summary) => ({
    id: `terminal-highlight-${summary?.outcome || 'unknown'}-${summary?.terminalTimestep || 0}`,
    beforeDatasetsDraw(chart) {
      const highlightStart = summary?.highlightStart;
      const highlightEnd = summary?.highlightEnd;
      if (!highlightStart || !highlightEnd) return;
      const xScale = chart.scales?.x;
      const chartArea = chart.chartArea;
      const labels = chart.data?.labels || [];
      if (!xScale || !chartArea || labels.length === 0) return;

      const startIndex = Math.max(0, Math.min(labels.length - 1, highlightStart - 1));
      const endIndex = Math.max(0, Math.min(labels.length - 1, highlightEnd - 1));
      const firstPixel = xScale.getPixelForValue(startIndex);
      const lastPixel = xScale.getPixelForValue(endIndex);
      const nextPixel = endIndex < labels.length - 1 ? xScale.getPixelForValue(endIndex + 1) : lastPixel;
      const prevPixel = startIndex > 0 ? xScale.getPixelForValue(startIndex - 1) : firstPixel;
      const stepHalfWidth = Math.max(8, Math.abs(nextPixel - prevPixel) / 2);

      const { ctx } = chart;
      ctx.save();
      ctx.fillStyle = summary.shadeColor;
      ctx.fillRect(
        firstPixel - stepHalfWidth,
        chartArea.top,
        (lastPixel - firstPixel) + stepHalfWidth * 2,
        chartArea.bottom - chartArea.top
      );
      ctx.strokeStyle = summary.accentColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(lastPixel, chartArea.top);
      ctx.lineTo(lastPixel, chartArea.bottom);
      ctx.stroke();
      ctx.restore();
    },
  }), []);

  const buildSelectedStepPlugin = useCallback((selectedStep) => ({
    id: `selected-step-${selectedStep || 0}`,
    afterDatasetsDraw(chart) {
      if (!selectedStep) return;
      const xScale = chart.scales?.x;
      const chartArea = chart.chartArea;
      const labels = chart.data?.labels || [];
      if (!xScale || !chartArea || labels.length === 0) return;
      const targetIndex = Math.max(0, Math.min(labels.length - 1, selectedStep - 1));
      const targetPixel = xScale.getPixelForValue(targetIndex);
      const { ctx } = chart;
      ctx.save();
      ctx.strokeStyle = 'rgba(15, 23, 42, 0.9)';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 6]);
      ctx.beginPath();
      ctx.moveTo(targetPixel, chartArea.top);
      ctx.lineTo(targetPixel, chartArea.bottom);
      ctx.stroke();
      ctx.restore();
    },
  }), []);

  const matchesEpisodeOutcome = useCallback((entry, filter) => {
    if (filter === 'all') return true;
    return (entry?.episode_outcome || 'unknown') === filter;
  }, []);

  const buildAverageEpisodeProfile = useCallback((entries, filter, labelPrefix) => {
    const eligibleEntries = entries.filter((entry) => Array.isArray(entry.reward_history) && entry.reward_history.length > 0);
    if (eligibleEntries.length === 0) {
      return null;
    }

    const histories = eligibleEntries.map((entry) => entry.reward_history);
    const maxLength = Math.max(...histories.map((history) => history.length), 0);
    const averageHistory = Array.from({ length: maxLength }, (_, index) => {
      const steps = histories.map((history) => history[index]).filter(Boolean);
      const breakdownKeys = Array.from(
        new Set(steps.flatMap((step) => Object.keys(step.reward_breakdown || {})))
      );
      const rawKeys = Array.from(
        new Set(steps.flatMap((step) => Object.keys(step.reward_raw_terms || {})))
      );

      const mean = (values) => {
        const valid = values.filter((value) => Number.isFinite(value));
        if (valid.length === 0) return null;
        return valid.reduce((sum, value) => sum + Number(value), 0) / valid.length;
      };

      const rewardBreakdown = {};
      for (const key of breakdownKeys) {
        rewardBreakdown[key] = mean(steps.map((step) => step.reward_breakdown?.[key]));
      }

      const rewardRawTerms = {};
      for (const key of rawKeys) {
        rewardRawTerms[key] = mean(steps.map((step) => step.reward_raw_terms?.[key]));
      }

      const rewardValue = mean(steps.map((step) => step.reward));
      return {
        step: index + 1,
        reward: rewardValue,
        reward_breakdown: rewardBreakdown,
        reward_raw_terms: rewardRawTerms,
      };
    });

    const averageTerminalTimestep = Math.round(
      eligibleEntries.reduce((sum, entry) => {
        const historyLength = Array.isArray(entry.reward_history) ? entry.reward_history.length : 0;
        return sum + Number(entry.episode_terminal_timestep || historyLength || 0);
      }, 0) / eligibleEntries.length
    );

    return {
      episode: `${labelPrefix}-${filter}-average`,
      reward: averageHistory.reduce((sum, step) => sum + Number(step.reward || 0), 0),
      reward_breakdown: averageHistory.reduce((acc, step) => {
        for (const [key, value] of Object.entries(step.reward_breakdown || {})) {
          acc[key] = (acc[key] || 0) + Number(value || 0);
        }
        return acc;
      }, {}),
      reward_history: averageHistory,
      episode_outcome: filter === 'all' ? 'unknown' : filter,
      episode_outcome_reason: `Average across ${eligibleEntries.length} ${filter === 'all' ? 'episodes' : `${filter} episodes`}.`,
      episode_terminal_timestep: averageTerminalTimestep,
      sample_count: eligibleEntries.length,
      is_average_profile: true,
    };
  }, []);

  const trainingGraphDefinitions = useMemo(() => {
    const orderedTicks = [...trainingRollouts].reverse();
    const labels = orderedTicks.map((entry) => entry.step);
    const latestBreakdownKeys = Object.keys(trainingRewardBreakdown || {}).filter((key) => key !== 'total');
    const meanBreakdownKeys = Object.keys(trainingRewardBreakdownMean || {}).filter((key) => key !== 'total');
    const breakdownKeys = Array.from(new Set([...latestBreakdownKeys, ...meanBreakdownKeys]));

    const makeSeries = (title, accessor, color) => ({
      title,
      labels,
      datasets: [
        {
          label: title,
          data: orderedTicks.map((entry) => accessor(entry)),
          borderColor: color,
          backgroundColor: color,
        },
      ],
    });

    return [
      makeSeries('Training Eval Reward', (entry) => entry.evalReward ?? null, 'rgb(16, 185, 129)'),
      makeSeries('Training Reward Mean', (entry) => entry.rewardMean ?? null, 'rgb(59, 130, 246)'),
      makeSeries('Training Reward (Latest Episode)', (entry) => entry.reward ?? null, 'rgb(249, 115, 22)'),
      makeSeries('Training Breakdown Total (Mean)', (entry) => entry.breakdownMean?.total ?? null, 'rgb(139, 92, 246)'),
      makeSeries('Training Breakdown Total (Latest)', (entry) => entry.breakdown?.total ?? null, 'rgb(236, 72, 153)'),
      ...breakdownKeys.flatMap((key, index) => ([
        makeSeries(
          `Training Breakdown Mean: ${key}`,
          (entry) => entry.breakdownMean?.[key] ?? null,
          `hsl(${(index * 47 + 190) % 360} 72% 48%)`
        ),
        makeSeries(
          `Training Breakdown Latest: ${key}`,
          (entry) => entry.breakdown?.[key] ?? null,
          `hsl(${(index * 47 + 20) % 360} 78% 56%)`
        ),
      ])),
    ];
  }, [trainingRewardBreakdown, trainingRewardBreakdownMean, trainingRollouts]);

  const currentTrainingGraph = trainingGraphDefinitions[trainingGraphIndex] || {
    title: 'Training Reward',
    labels: [],
    datasets: [],
  };

  const filteredTrainingEpisodes = useMemo(
    () => trainingEpisodes.filter((entry) => matchesEpisodeOutcome(entry, trainingTimelineOutcomeFilter)),
    [matchesEpisodeOutcome, trainingEpisodes, trainingTimelineOutcomeFilter]
  );

  const orderedFilteredTrainingEpisodes = useMemo(
    () => [...filteredTrainingEpisodes].sort((left, right) => Number(left?.episode ?? 0) - Number(right?.episode ?? 0)),
    [filteredTrainingEpisodes]
  );

  const selectedTrainingTimelineRollout = useMemo(() => {
    if (orderedFilteredTrainingEpisodes.length === 0) return null;
    if (selectedTrainingTimelineEpisode === null || selectedTrainingTimelineEpisode === undefined) {
      return orderedFilteredTrainingEpisodes[0];
    }
    return orderedFilteredTrainingEpisodes.find((entry) => entry.episode === selectedTrainingTimelineEpisode) || orderedFilteredTrainingEpisodes[0];
  }, [orderedFilteredTrainingEpisodes, selectedTrainingTimelineEpisode]);

  const selectedTrainingTimelineTarget = useMemo(() => {
    if (trainingTimelineMode === 'average') {
      return buildAverageEpisodeProfile(filteredTrainingEpisodes, trainingTimelineOutcomeFilter, 'training');
    }
    return selectedTrainingTimelineRollout;
  }, [buildAverageEpisodeProfile, filteredTrainingEpisodes, selectedTrainingTimelineRollout, trainingTimelineMode, trainingTimelineOutcomeFilter]);

  const trainingTimelineGraphDefinitions = useMemo(() => {
    const rewardHistory = Array.isArray(selectedTrainingTimelineTarget?.reward_history)
      ? selectedTrainingTimelineTarget.reward_history
      : [];
    const labels = rewardHistory.map((entry, index) => entry.step ?? index + 1);
    const breakdownKeys = Array.from(
      new Set(
        rewardHistory.flatMap((entry) =>
          Object.keys(entry.reward_breakdown || {}).filter((key) => key !== 'total')
        )
      )
    );
    const rawKeys = Array.from(
      new Set(
        rewardHistory.flatMap((entry) => Object.keys(entry.reward_raw_terms || {}))
      )
    );

    const makeSeries = (title, accessor, color) => ({
      title,
      labels,
      datasets: [
        {
          label: title,
          data: rewardHistory.map((entry) => accessor(entry)),
          borderColor: color,
          backgroundColor: color,
        },
      ],
    });

    return [
      {
        title: trainingTimelineMode === 'average'
          ? `Average Shaped Terms In ${trainingTimelineOutcomeFilter === 'all' ? 'All' : trainingTimelineOutcomeFilter.charAt(0).toUpperCase() + trainingTimelineOutcomeFilter.slice(1)} Training Episodes`
          : 'All Shaped Terms In Training Episode',
        labels,
        datasets: [
          {
            label: 'total_reward',
            data: rewardHistory.map((entry) => entry.reward_breakdown?.total ?? entry.reward ?? null),
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgb(59, 130, 246)',
          },
          ...breakdownKeys.map((key, index) => ({
            label: key,
            data: rewardHistory.map((entry) => entry.reward_breakdown?.[key] ?? null),
            borderColor: `hsl(${(index * 47 + 145) % 360} 72% 48%)`,
            backgroundColor: `hsl(${(index * 47 + 145) % 360} 72% 48%)`,
          })),
        ],
      },
      makeSeries(
        trainingTimelineMode === 'average' ? 'Average Training Episode Total Reward' : 'Training Episode Total Reward',
        (entry) => entry.reward_breakdown?.total ?? entry.reward ?? null,
        'rgb(59, 130, 246)'
      ),
      ...breakdownKeys.map((key, index) =>
        makeSeries(
          `Training Episode Term: ${key}`,
          (entry) => entry.reward_breakdown?.[key] ?? null,
          `hsl(${(index * 47 + 145) % 360} 72% 48%)`
        )
      ),
      ...rawKeys.map((key, index) =>
        makeSeries(
          `Training Episode Raw Term: ${key}`,
          (entry) => entry.reward_raw_terms?.[key] ?? null,
          `hsl(${(index * 47 + 305) % 360} 70% 45%)`
        )
      ),
    ];
  }, [selectedTrainingTimelineTarget, trainingTimelineMode, trainingTimelineOutcomeFilter]);

  const currentTrainingTimelineGraph = trainingTimelineGraphDefinitions[trainingTimelineGraphIndex] || {
    title: 'Training Episode Reward Timeline',
    labels: [],
    datasets: [],
  };
  const shouldShowTrainingTimelineKey = currentTrainingTimelineGraph.datasets.length > 1;
  const selectedTrainingTimelineSummary = useMemo(
    () => summarizeEpisodeOutcome(selectedTrainingTimelineTarget),
    [selectedTrainingTimelineTarget, summarizeEpisodeOutcome]
  );
  const trainingTimelinePlugins = useMemo(
    () => [buildTerminalHighlightPlugin(selectedTrainingTimelineSummary)],
    [buildTerminalHighlightPlugin, selectedTrainingTimelineSummary]
  );

  const rolloutGraphDefinitions = useMemo(() => {
    const orderedRollouts = [...rollouts].reverse();
    const labels = orderedRollouts.map((entry) => entry.episode);
    const breakdownKeys = Array.from(
      new Set(
        orderedRollouts.flatMap((entry) =>
          Object.keys(entry.reward_breakdown || {}).filter((key) => key !== 'total')
        )
      )
    );

    const makeSeries = (title, accessor, color) => ({
      title,
      labels,
      datasets: [
        {
          label: title,
          data: orderedRollouts.map((entry) => accessor(entry)),
          borderColor: color,
          backgroundColor: color,
        },
      ],
    });

    return [
      makeSeries('Rollout Total Reward', (entry) => entry.reward ?? null, 'rgb(56, 189, 248)'),
      makeSeries('Rollout Breakdown Total', (entry) => entry.reward_breakdown?.total ?? null, 'rgb(139, 92, 246)'),
      makeSeries('Rollout Native Reward', (entry) => entry.reward_breakdown?.native ?? entry.reward_raw_terms?.native ?? null, 'rgb(16, 185, 129)'),
      ...breakdownKeys.map((key, index) =>
        makeSeries(
          `Rollout Breakdown: ${key}`,
          (entry) => entry.reward_breakdown?.[key] ?? null,
          `hsl(${(index * 53 + 25) % 360} 74% 52%)`
        )
      ),
      ...breakdownKeys.map((key, index) =>
        makeSeries(
          `Rollout Raw Term: ${key}`,
          (entry) => entry.reward_raw_terms?.[key] ?? null,
          `hsl(${(index * 53 + 205) % 360} 70% 45%)`
        )
      ),
    ];
  }, [rollouts]);

  const currentRolloutGraph = rolloutGraphDefinitions[rolloutGraphIndex] || {
    title: 'Rollout Reward',
    labels: [],
    datasets: [],
  };

  const filteredRollouts = useMemo(
    () => rollouts.filter((entry) => matchesEpisodeOutcome(entry, rolloutTimelineOutcomeFilter)),
    [matchesEpisodeOutcome, rolloutTimelineOutcomeFilter, rollouts]
  );

  const orderedFilteredRollouts = useMemo(
    () => [...filteredRollouts].sort((left, right) => Number(left?.episode ?? 0) - Number(right?.episode ?? 0)),
    [filteredRollouts]
  );

  const earliestFilteredRollout = useMemo(() => {
    if (filteredRollouts.length === 0) return null;
    return filteredRollouts.reduce((earliest, entry) => {
      if (!earliest) return entry;
      const currentEpisode = Number(entry?.episode);
      const earliestEpisode = Number(earliest?.episode);
      if (!Number.isFinite(currentEpisode)) return earliest;
      if (!Number.isFinite(earliestEpisode) || currentEpisode < earliestEpisode) return entry;
      return earliest;
    }, null);
  }, [filteredRollouts]);

  const selectedTimelineRollout = useMemo(() => {
    if (orderedFilteredRollouts.length === 0) return null;
    if (selectedTimelineEpisode === null || selectedTimelineEpisode === undefined) {
      return earliestFilteredRollout || orderedFilteredRollouts[0];
    }
    return orderedFilteredRollouts.find((entry) => entry.episode === selectedTimelineEpisode) || earliestFilteredRollout || orderedFilteredRollouts[0];
  }, [earliestFilteredRollout, orderedFilteredRollouts, selectedTimelineEpisode]);

  const selectedTimelineTarget = useMemo(() => {
    if (rolloutTimelineMode === 'average') {
      return buildAverageEpisodeProfile(filteredRollouts, rolloutTimelineOutcomeFilter, 'rollout');
    }
    return selectedTimelineRollout;
  }, [buildAverageEpisodeProfile, filteredRollouts, rolloutTimelineMode, rolloutTimelineOutcomeFilter, selectedTimelineRollout]);

  const selectedEpisodeFrames = useMemo(() => {
    if (rolloutTimelineMode !== 'episode' || !selectedTimelineRollout) return [];
    return capturedEpisodeFramesByEpisode[selectedTimelineRollout.episode] || [];
  }, [capturedEpisodeFramesByEpisode, rolloutTimelineMode, selectedTimelineRollout]);

  const selectedTimelineStepEntry = useMemo(() => {
    if (!selectedTimelineTarget || !Array.isArray(selectedTimelineTarget.reward_history)) return null;
    if (!selectedTimelineStep) return null;
    return selectedTimelineTarget.reward_history[selectedTimelineStep - 1] || null;
  }, [selectedTimelineStep, selectedTimelineTarget]);

  const linkedSelectedFrameIndex = useMemo(() => {
    if (selectedEpisodeFrames.length === 0) return 0;
    if (!selectedTimelineStep) return 0;
    return Math.max(0, Math.min(selectedEpisodeFrames.length - 1, selectedTimelineStep - 1));
  }, [selectedEpisodeFrames, selectedTimelineStep]);

  const timelineGraphDefinitions = useMemo(() => {
    const rewardHistory = Array.isArray(selectedTimelineTarget?.reward_history)
      ? selectedTimelineTarget.reward_history
      : [];
    const labels = rewardHistory.map((entry, index) => entry.step ?? index + 1);
    const breakdownKeys = Array.from(
      new Set(
        rewardHistory.flatMap((entry) =>
          Object.keys(entry.reward_breakdown || {}).filter((key) => key !== 'total')
        )
      )
    );
    const rawKeys = Array.from(
      new Set(
        rewardHistory.flatMap((entry) =>
          Object.keys(entry.reward_raw_terms || {})
        )
      )
    );

    const makeSeries = (title, accessor, color) => ({
      title,
      labels,
      datasets: [
        {
          label: title,
          data: rewardHistory.map((entry) => accessor(entry)),
          borderColor: color,
          backgroundColor: color,
        },
      ],
    });

    return [
      {
        title: rolloutTimelineMode === 'average'
          ? `Average Shaped Terms In ${rolloutTimelineOutcomeFilter === 'all' ? 'All' : rolloutTimelineOutcomeFilter.charAt(0).toUpperCase() + rolloutTimelineOutcomeFilter.slice(1)} Episodes`
          : 'All Shaped Terms In Episode',
        labels,
        datasets: [
          {
            label: 'total_reward',
            data: rewardHistory.map((entry) => entry.reward_breakdown?.total ?? entry.reward ?? null),
            borderColor: 'rgb(56, 189, 248)',
            backgroundColor: 'rgb(56, 189, 248)',
          },
          ...breakdownKeys.map((key, index) => ({
            label: key,
            data: rewardHistory.map((entry) => entry.reward_breakdown?.[key] ?? null),
            borderColor: `hsl(${(index * 47 + 25) % 360} 74% 52%)`,
            backgroundColor: `hsl(${(index * 47 + 25) % 360} 74% 52%)`,
          })),
        ],
      },
      makeSeries(
        rolloutTimelineMode === 'average' ? 'Average Episode Total Reward' : 'Episode Total Reward',
        (entry) => entry.reward_breakdown?.total ?? entry.reward ?? null,
        'rgb(56, 189, 248)'
      ),
      ...breakdownKeys.map((key, index) =>
        makeSeries(
          `Episode Term: ${key}`,
          (entry) => entry.reward_breakdown?.[key] ?? null,
          `hsl(${(index * 47 + 25) % 360} 74% 52%)`
        )
      ),
      ...rawKeys.map((key, index) =>
        makeSeries(
          `Episode Raw Term: ${key}`,
          (entry) => entry.reward_raw_terms?.[key] ?? null,
          `hsl(${(index * 47 + 205) % 360} 70% 45%)`
        )
      ),
    ];
  }, [rolloutTimelineMode, rolloutTimelineOutcomeFilter, selectedTimelineTarget]);

  const currentTimelineGraph = timelineGraphDefinitions[timelineGraphIndex] || {
    title: 'Episode Reward Timeline',
    labels: [],
    datasets: [],
  };
  const shouldShowRolloutTimelineKey = currentTimelineGraph.datasets.length > 1;
  const selectedRolloutTimelineSummary = useMemo(
    () => summarizeEpisodeOutcome(selectedTimelineTarget),
    [selectedTimelineTarget, summarizeEpisodeOutcome]
  );
  const rolloutTimelinePlugins = useMemo(
    () => [
      buildTerminalHighlightPlugin(selectedRolloutTimelineSummary),
      buildSelectedStepPlugin(selectedTimelineStep),
    ],
    [buildSelectedStepPlugin, buildTerminalHighlightPlugin, selectedRolloutTimelineSummary, selectedTimelineStep]
  );
  const trainingOutcomeCounts = useMemo(() => ({
    success: trainingEpisodes.filter((entry) => entry.episode_outcome === 'success').length,
    failure: trainingEpisodes.filter((entry) => entry.episode_outcome === 'failure').length,
  }), [trainingEpisodes]);
  const trainingWorkspaceViews = useMemo(
    () => ([
      { key: 'reward_chart', title: 'Training Reward Chart' },
      { key: 'temporal_breakdown', title: 'Training Temporal Breakdown' },
    ]),
    []
  );
  const currentTrainingWorkspaceView = trainingWorkspaceViews[trainingWorkspaceViewIndex] || trainingWorkspaceViews[0];
  const rolloutOutcomeCounts = useMemo(() => ({
    success: rollouts.filter((entry) => entry.episode_outcome === 'success').length,
    failure: rollouts.filter((entry) => entry.episode_outcome === 'failure').length,
  }), [rollouts]);
  const rolloutWorkspaceViews = useMemo(
    () => ([
      { key: 'visualization', title: 'Rollout Visualization' },
      { key: 'reward_chart', title: 'Rollout Reward Chart' },
      { key: 'temporal_breakdown', title: 'Temporal Reward Breakdown' },
    ]),
    []
  );
  const currentRolloutWorkspaceView = rolloutWorkspaceViews[rolloutWorkspaceViewIndex] || rolloutWorkspaceViews[0];
  const availableVisualizationEpisodes = useMemo(
    () => Object.keys(capturedEpisodeFramesByEpisode)
      .map((key) => Number(key))
      .filter((value) => Number.isFinite(value))
      .sort((a, b) => a - b),
    [capturedEpisodeFramesByEpisode]
  );
  const visualizationFrames = useMemo(() => {
    if (selectedVisualizationEpisode !== null && selectedVisualizationEpisode !== undefined) {
      return capturedEpisodeFramesByEpisode[selectedVisualizationEpisode] || [];
    }
    return frames || [];
  }, [capturedEpisodeFramesByEpisode, frames, selectedVisualizationEpisode]);

  const saveRewardConfig = useCallback(async () => {
    if (isSavedViewer) {
      setRewardConfigStatus("Saved rollout viewers are read-only. Apply reward changes from a live rollout.");
      return;
    }
    if (!runId) {
      console.warn("Run ID not set yet, cannot save reward config");
      return;
    }

    setRewardConfigLoading(true);
    try {
      // Reward config is being generated from the current reward config terms, so we send the current terms to ensure the backend has the latest version of the config (in case there are unsaved changes) and can validate it before applying.
      const response = await apiClient.post("/reward_config", {
        run_id: runId,
        env_name: envName,
        terms: rewardConfig.map((term) => ({
          key: term.key,
          label: term.label,
          description: term.description,
          weight: Number(term.weight),
          enabled: Boolean(term.enabled),
          expression: term.expression || '',
        })),
        goal: taskProposal?.goal || taskGoal || undefined,
        task_params: taskProposal?.task_params || [],
        derived_signals: taskProposal?.derived_signals || [],
        success_metric: taskProposal?.success_metric || '',
        rationale: taskProposal?.rationale || '',
        warnings: taskProposal?.warnings || [],
        provider: taskProposal?.provider || undefined,
        model: taskProposal?.model || undefined,
        behavior_plan: taskProposal?.behavior_plan || {},
      });
      const nextTerms = response.data.terms || [];
      setRewardConfig(nextTerms);
      setAvailableRewardVariables(response.data.available_variables || []);
      setRewardSourceLinks(response.data.reward_source_links || []);
      setSavedRewardConfigFiles(response.data.saved_reward_configs || []);
      setSelectedRewardConfigFile((prev) => (prev && (response.data.saved_reward_configs || []).includes(prev) ? prev : ((response.data.saved_reward_configs || [])[0] || '')));
      setTrainingEpisodes((prev) => applyRewardConfigToTrainingEpisodes(prev, nextTerms));
      const nextRollouts = applyRewardConfigToRollouts(rollouts, nextTerms);
      setRollouts(nextRollouts);
      const nextBreakdown = computeBreakdownFromRawTerms(nextTerms, latestRolloutRawTerms);
      if (nextRollouts.length > 0) {
        setEpisodeInfo((prev) => ({ ...prev, reward: nextRollouts[0].reward ?? prev.reward }));
      } else if (nextBreakdown) {
        setEpisodeInfo((prev) => ({ ...prev, reward: nextBreakdown.total }));
      }
      if (nextRollouts.length > 0) {
        setRolloutRewardBreakdown(nextRollouts[0].reward_breakdown || {});
      } else if (nextBreakdown) {
        setRolloutRewardBreakdown(nextBreakdown);
      }
      setRewardConfigDirty(false);
      setRewardConfigStatus("Reward settings applied live.");
    } catch (error) {
      console.error("Failed to update reward config:", error);
      const backendMessage =
        error?.response?.data?.detail ||
        error?.response?.data?.error ||
        error?.message ||
        "Failed to save reward settings.";
      setRewardConfigStatus(String(backendMessage));
    } finally {
      setRewardConfigLoading(false);
    }
  }, [applyRewardConfigToRollouts, applyRewardConfigToTrainingEpisodes, computeBreakdownFromRawTerms, envName, isSavedViewer, latestRolloutRawTerms, rewardConfig, rollouts, runId, taskGoal, taskProposal]);

  const saveRewardConfigSnapshot = useCallback(async () => {
    if (isSavedViewer) {
      setRewardConfigStatus("Saved rollout viewers are read-only. Save reward configs from a live rollout.");
      return;
    }
    if (!runId) {
      setRewardConfigStatus("Run ID not ready yet. Wait a moment and try again.");
      return;
    }
    setRewardConfigLoading(true);
    try {
      const response = await apiClient.post('/save_reward_config', {
        run_id: runId,
        env_name: envName,
        filename: rewardConfigSaveName || undefined,
        source_type: rewardConfigSaveSourceType || undefined,
      });
      const files = response.data.saved_reward_configs || [];
      setSavedRewardConfigFiles(files);
      setSelectedRewardConfigFile(response.data.filename || files[0] || '');
      if (!rewardConfigSaveName && response.data.filename) {
        setRewardConfigSaveName(response.data.filename);
      }
      setRewardConfigStatus(`Saved reward config to ${response.data.filename}.`);
    } catch (error) {
      console.error('Failed to save reward config snapshot:', error);
      const backendMessage =
        error?.response?.data?.detail ||
        error?.response?.data?.error ||
        error?.message ||
        'Failed to save reward config snapshot.';
      setRewardConfigStatus(String(backendMessage));
    } finally {
      setRewardConfigLoading(false);
    }
  }, [envName, isSavedViewer, rewardConfigSaveName, rewardConfigSaveSourceType, runId]);

  const loadSavedRewardConfig = useCallback(async () => {
    if (isSavedViewer) {
      setRewardConfigStatus("Saved rollout viewers are read-only. Load reward configs from a live rollout.");
      return;
    }
    if (!runId) {
      setRewardConfigStatus("Run ID not ready yet. Wait a moment and try again.");
      return;
    }
    if (!selectedRewardConfigFile) {
      setRewardConfigStatus("Select a saved reward config first.");
      return;
    }
    setRewardConfigLoading(true);
    try {
      const response = await apiClient.post('/load_reward_config', {
        run_id: runId,
        env_name: envName,
        filename: selectedRewardConfigFile,
      });
      const nextTerms = response.data.terms || [];
      setRewardConfig(nextTerms);
      setSupportsCustomReward(Boolean(nextTerms.length > 1));
      setAvailableRewardVariables(response.data.available_variables || []);
      setRewardFormulaExamples(response.data.formula_examples || []);
      setRewardSourceLinks(response.data.reward_source_links || []);
      setAvailableBehaviorTags(response.data.available_behavior_tags || []);
      setSavedRewardConfigFiles(response.data.saved_reward_configs || []);
      setSelectedRewardConfigFile(response.data.filename || selectedRewardConfigFile);
      setRewardConfigSaveSourceType(response.data.source_type || 'manual');
      setTaskGoal(response.data.task_config?.goal || '');
      setTaskProposal((prev) => normalizeTaskProposal({
        ...(prev || {}),
        reward_terms: nextTerms,
        ...(response.data.task_config || {}),
      }));
      setTrainingEpisodes((prev) => applyRewardConfigToTrainingEpisodes(prev, nextTerms));
      const nextRollouts = applyRewardConfigToRollouts(rollouts, nextTerms);
      setRollouts(nextRollouts);
      const nextBreakdown = computeBreakdownFromRawTerms(nextTerms, latestRolloutRawTerms);
      if (nextRollouts.length > 0) {
        setEpisodeInfo((prev) => ({ ...prev, reward: nextRollouts[0].reward ?? prev.reward }));
        setRolloutRewardBreakdown(nextRollouts[0].reward_breakdown || {});
      } else if (nextBreakdown) {
        setEpisodeInfo((prev) => ({ ...prev, reward: nextBreakdown.total }));
        setRolloutRewardBreakdown(nextBreakdown);
      }
      setRewardConfigDirty(false);
      setRewardConfigStatus(`Loaded reward config ${response.data.filename}.`);
    } catch (error) {
      console.error('Failed to load reward config snapshot:', error);
      const backendMessage =
        error?.response?.data?.detail ||
        error?.response?.data?.error ||
        error?.message ||
        'Failed to load saved reward config.';
      setRewardConfigStatus(String(backendMessage));
    } finally {
      setRewardConfigLoading(false);
    }
  }, [applyRewardConfigToRollouts, applyRewardConfigToTrainingEpisodes, computeBreakdownFromRawTerms, envName, isSavedViewer, latestRolloutRawTerms, normalizeTaskProposal, rollouts, runId, selectedRewardConfigFile]);

  const proposeTaskConfig = useCallback(async () => {
    if (isSavedViewer) {
      showSavedViewerTaskMessage();
      return;
    }
    const trimmedGoal = String(taskGoal || '').trim();
    if (!trimmedGoal) {
      setTaskProposalStatus('Enter a natural-language task goal first.');
      return;
    }
    if (!runId) {
      setTaskProposalStatus('Run ID not ready yet. Wait a moment and try again.');
      return;
    }
    setTaskProposalLoading(true);
    setTaskProposalLiveStatus(null);
    setTaskProposalStatus('Submitting task-config proposal request...');
    try {
      const response = await apiClient.post('/propose_task_config', {
        run_id: runId,
        env_name: envName,
        goal: trimmedGoal,
        llm_id: selectedLlmId || undefined,
        proposal_strategy: proposalStrategy,
      });
      const normalizedProposal = normalizeTaskProposal(response.data);
      setTaskProposal(normalizedProposal);
      const previewTerms = buildProposalRewardPreviewTerms(normalizedProposal);
      if (previewTerms.length > 0) {
        setRewardConfig(previewTerms);
        setSupportsCustomReward(true);
        setRewardConfigDirty(true);
        setRewardConfigStatus('Proposal copied into Reward Terms locally. Click Apply Proposal to persist the task config and reward terms on the backend.');
      }
      setAvailableBehaviorTags(response.data.available_behavior_tags || []);
      {
        const nextLlms = response.data.available_llms || [];
        setAvailableLlms(nextLlms);
        setSelectedLlmId((prev) => {
          if (prev && nextLlms.some((item) => item.id === prev && item.available)) {
            return prev;
          }
          return response.data.default_llm_id || nextLlms.find((item) => item.available)?.id || '';
        });
      }
      setTaskProposalLiveStatus(null);
      const provider = response.data?.provider || 'proposal';
      setTaskProposalStatus(`Generated ${provider} task config proposal. Review it, then apply if it looks right.`);
    } catch (error) {
      console.error('Failed to propose task config:', error);
      const backendMessage =
        error?.response?.data?.detail ||
        error?.response?.data?.error ||
        error?.message ||
        'Failed to generate task config proposal.';
      setTaskProposalStatus(String(backendMessage));
    } finally {
      setTaskProposalLoading(false);
    }
  }, [buildProposalRewardPreviewTerms, envName, isSavedViewer, normalizeTaskProposal, proposalStrategy, runId, selectedLlmId, showSavedViewerTaskMessage, taskGoal]);

  useEffect(() => {
    if (isSavedViewer || !runId || !taskProposalLoading) return undefined;
    let cancelled = false;

    const pollStatus = async () => {
      try {
        const response = await apiClient.get(`/task_config_status/${runId}`);
        if (cancelled) return;
        setTaskProposalLiveStatus(response.data || null);
        if (response.data?.message) {
          const attemptText = response.data?.attempt ? ` [attempt ${response.data.attempt}]` : '';
          const elapsedText = Number.isFinite(response.data?.elapsed_sec) ? ` (${Number(response.data.elapsed_sec).toFixed(1)}s)` : '';
          setTaskProposalStatus(`${response.data.message}${attemptText}${elapsedText}`);
        }
      } catch (error) {
        if (cancelled) return;
      }
    };

    pollStatus();
    const interval = setInterval(pollStatus, 1000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [isSavedViewer, runId, taskProposalLoading]);

  const applyTaskProposal = useCallback(async () => {
    if (isSavedViewer) {
      showSavedViewerTaskMessage();
      return;
    }
    if (!taskProposal) {
      setTaskProposalStatus('No task proposal available yet.');
      return;
    }
    if (!runId) {
      setTaskProposalStatus('Run ID not ready yet. Wait a moment and try again.');
      return;
    }
    setTaskProposalLoading(true);
    try {
      const response = await apiClient.post('/apply_task_config', {
        run_id: runId,
        env_name: envName,
        goal: taskProposal.goal || taskGoal,
        task_params: taskProposal.task_params || [],
        derived_signals: taskProposal.derived_signals || [],
        reward_terms: (taskProposal.reward_terms || []).map((term) => ({
          key: term.key,
          label: term.label,
          description: term.description,
          weight: Number(term.weight),
          enabled: Boolean(term.enabled),
          expression: term.expression || '',
        })),
        success_metric: taskProposal.success_metric || '',
        rationale: taskProposal.rationale || '',
        warnings: taskProposal.warnings || [],
        provider: taskProposal.provider || 'manual',
        model: taskProposal.model || '',
        behavior_plan: taskProposal.behavior_plan || {},
      });
      const nextTerms = response.data.terms || [];
      setRewardConfig(nextTerms);
      setAvailableRewardVariables(response.data.available_variables || []);
      setRewardFormulaExamples(response.data.formula_examples || []);
      setRewardSourceLinks(response.data.reward_source_links || []);
      setAvailableBehaviorTags(response.data.available_behavior_tags || []);
      setSavedRewardConfigFiles(response.data.saved_reward_configs || []);
      setSelectedRewardConfigFile((prev) => (prev && (response.data.saved_reward_configs || []).includes(prev) ? prev : ((response.data.saved_reward_configs || [])[0] || '')));
      setRewardConfigSaveSourceType(response.data.source_type || 'manual');
      setTrainingEpisodes((prev) => applyRewardConfigToTrainingEpisodes(prev, nextTerms));
      const nextRollouts = applyRewardConfigToRollouts(rollouts, nextTerms);
      setRollouts(nextRollouts);
      const nextBreakdown = computeBreakdownFromRawTerms(nextTerms, latestRolloutRawTerms);
      if (nextRollouts.length > 0) {
        setEpisodeInfo((prev) => ({ ...prev, reward: nextRollouts[0].reward ?? prev.reward }));
        setRolloutRewardBreakdown(nextRollouts[0].reward_breakdown || {});
      } else if (nextBreakdown) {
        setEpisodeInfo((prev) => ({ ...prev, reward: nextBreakdown.total }));
        setRolloutRewardBreakdown(nextBreakdown);
      }
      setRewardConfigDirty(false);
      setRewardConfigStatus('Task proposal applied live.');
      setTaskGoal(response.data?.task_config?.goal || taskProposal.goal || taskGoal);
      setTaskProposal((prev) => normalizeTaskProposal({
        ...(prev || {}),
        reward_terms: nextTerms,
        ...(response.data?.task_config || {}),
      }));
      setTaskProposalStatus('Task proposal applied. Training and rollout now use the proposed reward config.');
    } catch (error) {
      console.error('Failed to apply task proposal:', error);
      const backendMessage =
        error?.response?.data?.detail ||
        error?.response?.data?.error ||
        error?.message ||
        'Failed to apply task config proposal.';
      setTaskProposalStatus(String(backendMessage));
    } finally {
      setTaskProposalLoading(false);
    }
  }, [
    applyRewardConfigToRollouts,
    applyRewardConfigToTrainingEpisodes,
    computeBreakdownFromRawTerms,
    envName,
    isSavedViewer,
    latestRolloutRawTerms,
    rollouts,
    runId,
    showSavedViewerTaskMessage,
    normalizeTaskProposal,
    taskGoal,
    taskProposal,
  ]);
  // This effectively flips the showPopup
  // showPopup = true => showPopup = false, and vice versa
  const togglePopup = () => {
    setShowPopup((prev) => !prev);
  };

  const [showPathPopup, setShowPathPopup] = useState(false);
  const [trainingPath, setTrainingPath] = useState("models/basic_model.zip");
  const [trainingHyperparams, setTrainingHyperparams] = useState({
    learning_rate: 0.0003,
    lr_schedule: "constant",
    n_steps: 2048,
    batch_size: 64,
    n_epochs: 10,
    gamma: 0.99,
    gae_lambda: 0.95,
    clip_range: 0.2,
    ent_coef: 0.0,
    vf_coef: 0.5,
    max_grad_norm: 0.5,
    model_size: "medium",
  });

  // basically triggers the /models POST request again so the models can be read again
  const [reloadAllTempModelsSwitcher, setReloadAllTempModelsSwitcher] = useState(false);
  const openPathPopup = () => setShowPathPopup(true);
  const closePathPopup = () => setShowPathPopup(false);

  const [frozenPath, setFrozenPath] = useState(null);

  //gets the frozen Path, so the default Path doesn't change too drastically. 
  useEffect(() => {
      if (showPathPopup && frozenPath === null) {
        setFrozenPath(`ppo_model_${envName}_${timestamp}.zip`);
      }
      if (!showPathPopup) {
        // reset so a new one is generated next time
        setFrozenPath(null);
      }
  }, [showPathPopup, envName, frozenPath]);

  // toggle train and pause at the same time
  const toggleTrainPauseTogether = () => {
    const newValue = !trainMode;
    const newPauseValue = trainMode;
    setTrainMode(newValue);
    // console.log("Toggling the pause value to:", newPauseValue); // DEBUG:FRONTEND
    togglePause(newPauseValue); // pause if train mode is toggled
  };

  const reloadTempModels = () => {
    if (!runId) {
      console.warn("Run ID not set yet, cannot pause/resume");
      return;
    }
    // should force useEffect to run again
    setReloadAllTempModelsSwitcher(prev => !prev);
  };

  const updateRolloutSpeed = async (newSpeed) => {
    setRolloutSpeed(newSpeed);

    try {
      await apiClient.post("/rollout_speed", { "run_id": runId, "fps":newSpeed,"delay": 1.0 / newSpeed });
    } catch (e) {
      console.error("Failed to set rollout speed:", e);
    }
  };
  const saveTrainingPath = async (path, device, nextTrainSteps, nextTrainingHyperparams) => {
    if (!runId) {
      console.warn("Run ID not set yet, cannot pause/resume");
      return;
    }
    try {
      // Set Train path FIRST
      // THEN SET THE TRAIN MODE AND toggle pause
      // persist to backend (example endpoint)
      await apiClient.post("/set_training_dir", {
        "run_id": runId,
        "train_dir_path": path,
        "device": device,
        "env_name": envName,
        "training_hyperparams": nextTrainingHyperparams,
      });
      setTrainingPath(path);
      setTrainSteps(Number(nextTrainSteps) || trainSteps);
      setReloadAllTempModelsSwitcher((prev) => !prev);
      setTrainingHyperparams(nextTrainingHyperparams);
      setTrainingAblationReport(null);
      setTrainingAblationStatus('idle');
      setTrainingInsights(null);
      setTrainingBehaviorReport(null);
      setTrainingBehaviorTags(null);
      setShowTrainingInsights(false);
      closePathPopup();

      toggleTrainPauseTogether();

      
    } catch (e) {
      console.error("Failed to set training path:", e);
      // optionally show a toast here
    }
  };
  const deleteAllTempModels = () => {
    // delete all the models in the temporary directory
    apiClient.post("/delete_all_temp_models", { "run_id": runId });
    // force reload of the temp model directory
    setReloadAllTempModelsSwitcher(prev => !prev);
    
  };
  const changeNumberOfSteps = async (steps) => {
    setStepInterval(steps);
    if (!runId) {
      console.warn("Run ID not set yet, cannot pause/resume");
      return;
    }
    try {
      await apiClient.post("/change_number_of_steps", { "run_id": runId, "number_of_steps": steps });
    } catch (e) {
      console.error("Failed to change number of steps:", e);
    }
  };

  const togglePause = async(ns) => {
    if (!sessionId) {
      console.warn("Session ID not set yet, cannot pause/resume");
      return;
    }
    const newState = ns !== undefined ? ns : !isPaused;
    // console.log("Sending pause state:", newState); // DEBUG:FRONTEND
    await apiClient.post("/pause_rollout", { session_id: sessionId, paused: newState });
    isPausedRef.current = newState;
    setIsPaused(newState);
  };

  useEffect(() => {
    if (isSavedViewer) {
      setRewardConfig([]);
      setSupportsCustomReward(false);
      setAvailableRewardVariables([]);
      setRewardFormulaExamples([]);
      setRewardSourceLinks([]);
      setTaskGoal('');
      setTaskProposal(null);
      setTaskProposalLoading(false);
      setTaskProposalStatus('Saved rollout viewer. Task config proposals are only available in live rollout windows.');
      setRewardConfigDirty(false);
      setRewardConfigLoading(false);
      setRewardConfigStatus("Saved rollout viewer. Reward terms shown below come from the loaded JSON.");
      setTrainingAblationReport(null);
      setTrainingAblationStatus('idle');
      setTrainingInsights(null);
      setRolloutInsights(null);
      setTrainingBehaviorReport(null);
      setTrainingBehaviorTags(null);
      setRolloutBehaviorReport(null);
      setRolloutBehaviorTags(null);
      setShowTrainingInsights(false);
      setShowRolloutInsights(false);
      hydrateLoadedRollouts(initialRollouts);
    }
  }, [hydrateLoadedRollouts, initialRollouts, isSavedViewer]);

  useEffect(() => {
    if (trainingGraphDefinitions.length === 0) {
      setTrainingGraphIndex(0);
      return;
    }
    setTrainingGraphIndex((prev) => prev % trainingGraphDefinitions.length);
  }, [trainingGraphDefinitions.length]);

  useEffect(() => {
    if (trainingTimelineGraphDefinitions.length === 0) {
      setTrainingTimelineGraphIndex(0);
      return;
    }
    setTrainingTimelineGraphIndex((prev) => prev % trainingTimelineGraphDefinitions.length);
  }, [trainingTimelineGraphDefinitions.length]);

  useEffect(() => {
    if (filteredTrainingEpisodes.length === 0) {
      setSelectedTrainingTimelineEpisode(null);
      return;
    }
    const hasSelectedEpisode = filteredTrainingEpisodes.some((entry) => entry.episode === selectedTrainingTimelineEpisode);
    if (!hasSelectedEpisode) {
      setSelectedTrainingTimelineEpisode(filteredTrainingEpisodes[0].episode ?? null);
    }
  }, [filteredTrainingEpisodes, selectedTrainingTimelineEpisode]);

  useEffect(() => {
    if (availableVisualizationEpisodes.length === 0) {
      setSelectedVisualizationEpisode(null);
      return;
    }
    const hasSelectedEpisode = availableVisualizationEpisodes.includes(Number(selectedVisualizationEpisode));
    if (!hasSelectedEpisode) {
      setSelectedVisualizationEpisode(availableVisualizationEpisodes[0]);
    }
  }, [availableVisualizationEpisodes, selectedVisualizationEpisode]);

  useEffect(() => {
    setCurrentFrame(0);
    setIsPlaying(false);
  }, [selectedVisualizationEpisode]);

  useEffect(() => {
    if (rolloutGraphDefinitions.length === 0) {
      setRolloutGraphIndex(0);
      return;
    }
    setRolloutGraphIndex((prev) => prev % rolloutGraphDefinitions.length);
  }, [rolloutGraphDefinitions.length]);

  useEffect(() => {
    if (timelineGraphDefinitions.length === 0) {
      setTimelineGraphIndex(0);
      return;
    }
    setTimelineGraphIndex((prev) => prev % timelineGraphDefinitions.length);
  }, [timelineGraphDefinitions.length]);

  useEffect(() => {
    if (filteredRollouts.length === 0) {
      setSelectedTimelineEpisode(null);
      setSelectedTimelineStep(null);
      return;
    }
    const hasSelectedEpisode = filteredRollouts.some((entry) => entry.episode === selectedTimelineEpisode);
    if (!hasSelectedEpisode) {
      setSelectedTimelineEpisode(earliestFilteredRollout?.episode ?? filteredRollouts[0].episode ?? null);
    }
  }, [earliestFilteredRollout, filteredRollouts, selectedTimelineEpisode]);

  useEffect(() => {
    if (!selectedTimelineTarget || !Array.isArray(selectedTimelineTarget.reward_history) || selectedTimelineTarget.reward_history.length === 0) {
      setSelectedTimelineStep(null);
      return;
    }
    const maxStep = selectedTimelineTarget.reward_history.length;
    const defaultStep = Math.min(
      Number(selectedTimelineTarget.episode_terminal_timestep || maxStep),
      maxStep
    );
    if (!selectedTimelineStep || selectedTimelineStep > maxStep) {
      setSelectedTimelineStep(defaultStep);
    }
  }, [selectedTimelineStep, selectedTimelineTarget]);

  useEffect(() => {
    if (rolloutTimelineMode !== 'episode') return;
    if (selectedEpisodeFrames.length === 0) return;
    if (!selectedTimelineStep) return;
    const nextFrameIndex = Math.max(0, Math.min(selectedEpisodeFrames.length - 1, selectedTimelineStep - 1));
    setCurrentFrame((prev) => (prev === nextFrameIndex ? prev : nextFrameIndex));
  }, [rolloutTimelineMode, selectedEpisodeFrames, selectedTimelineStep]);

  useEffect(() => {
    let retryTimeout;
    // console.log("Fetching models from server..."); // DEBUG:FRONTEND
    const fetchModels = async () => {
      try {
        const res = await apiClient.get("/models");
        const modelNames = Array.isArray(res.data?.models) ? res.data.models : [];
        const modelRecords = Array.isArray(res.data?.model_records) ? res.data.model_records : [];
        setServerModels(modelNames);
        setServerModelRecords(modelRecords);
        setSelectedServerModel((prev) => {
          if (prev && modelNames.includes(prev)) return prev;
          return modelNames[0] || "";
        });
      } catch (e) {
        console.error("List the models process has failed: Will retry in 5 seconds");
        //retry timeout = 5 seconds
        retryTimeout = setTimeout(fetchModels, 5000);
      }
    };
    const fetchRolloutFiles = async () => {
      try {
        const res = await apiClient.get("/rollouts_files");
        const files = res.data.rollouts || [];
        setRolloutFiles(files);
        setSelectedRolloutFile((prev) => (prev && files.includes(prev) ? prev : files[0] || ""));
      } catch (e) {
        console.error("Failed to list saved rollouts:", e);
      }
    };
    fetchModels();
    fetchRolloutFiles();
    return () => {
      if (retryTimeout) clearTimeout(retryTimeout);
    }
  }, [reloadAllTempModelsSwitcher]);
  useEffect(() => {
    if (isSavedViewer) return;
    let cancelled = false;
    let retryTimer = null;
    let attempt = 0;

    async function fetchRunId() {
      try {
        const returnData = await apiClient.get("/unique_run_id");
        if (!cancelled) {
          setRunId(returnData.data.run_id);
          attempt = 0; // reset the backoff attempt after success
        }
      } catch (error) {
        if (!cancelled) {
          // if intential cancel, don't retry
          if (isCancel?.(error) || error?.name === "CanceledError") return;
          attempt ++;
          const delay = Math.min(30000, 1000 * 2 ** attempt); // exponential backoff up to 30s
          retryTimer  = setTimeout(fetchRunId, delay);
          console.warn(`run_id fetch failed (attempt ${attempt}), retrying in ${delay}ms`);

        }
        console.error("Error fetching unique run ID:", error);
      }
      
    }
    fetchRunId();
    // console.log("Run id current: ", runId); // DEBUG:FRONTEND
    return () => {
      // run when cancelled
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
    }
  }, []);

  useEffect(() => {
    if (isSavedViewer || !runId) return;

    let cancelled = false;

    async function fetchRewardConfig() {
      setRewardConfigLoading(true);
      try {
        const response = await apiClient.get("/reward_config", {
          params: { run_id: runId, env_name: envName },
        });
        if (cancelled) return;
        setRewardConfig(response.data.terms || []);
        setSupportsCustomReward(Boolean(response.data.supports_custom_reward));
        setAvailableRewardVariables(response.data.available_variables || []);
        setRewardFormulaExamples(response.data.formula_examples || []);
        setRewardSourceLinks(response.data.reward_source_links || []);
        setAvailableBehaviorTags(response.data.available_behavior_tags || []);
        const nextLlms = response.data.available_llms || [];
        setAvailableLlms(nextLlms);
        setSelectedLlmId((prev) => {
          if (prev && nextLlms.some((item) => item.id === prev && item.available)) {
            return prev;
          }
          return response.data.default_llm_id || nextLlms.find((item) => item.available)?.id || '';
        });
        setSavedRewardConfigFiles(response.data.saved_reward_configs || []);
        setSelectedRewardConfigFile((prev) => (prev && (response.data.saved_reward_configs || []).includes(prev) ? prev : ((response.data.saved_reward_configs || [])[0] || '')));
        setRewardConfigSaveSourceType(response.data.source_type || 'manual');
        setTaskGoal(response.data.task_config?.goal || '');
        setTaskProposal((prev) => normalizeTaskProposal({
          ...(prev || {}),
          ...(response.data.task_config || {}),
          available_behavior_tags: response.data.available_behavior_tags || [],
        }));
        setRewardConfigDirty(false);
        setRewardConfigStatus(
          response.data.supports_custom_reward
            ? "Editing applies to rollout and training live."
            : "Only native Gym reward is available for this environment right now."
        );
        setTaskProposalStatus(
          response.data.task_config?.goal
            ? 'Task goal restored for this run. Generate a new proposal or apply updated reward edits.'
            : 'Describe a task goal, then generate a proposal.'
        );
      } catch (error) {
        if (cancelled) return;
        console.error("Failed to fetch reward config:", error);
        setAvailableRewardVariables([]);
        setRewardFormulaExamples([]);
        setRewardSourceLinks([]);
        setAvailableBehaviorTags([]);
        setTaskGoal('');
        setTaskProposal(null);
        setRewardConfigStatus("Failed to load reward settings.");
        setTaskProposalStatus('Failed to load task config state.');
      } finally {
        if (!cancelled) {
          setRewardConfigLoading(false);
        }
      }
    }

    fetchRewardConfig();
    return () => {
      cancelled = true;
    };
  }, [envName, isSavedViewer, runId]);

  useEffect(() => {
    if (isSavedViewer) return undefined;
    let cancelled = false;

    async function fetchTaskConfigLlms() {
      try {
        const response = await apiClient.get('/task_config_llms');
        if (cancelled) return;
        const nextLlms = response.data?.llms || [];
        setAvailableLlms(nextLlms);
        setSelectedLlmId((prev) => {
          if (prev && nextLlms.some((item) => item.id === prev && item.available)) {
            return prev;
          }
          return response.data?.default_llm_id || nextLlms.find((item) => item.available)?.id || '';
        });
      } catch (error) {
        if (cancelled) return;
        console.error('Failed to fetch task-config LLMs:', error);
        setAvailableLlms([]);
        setSelectedLlmId('');
      }
    }

    fetchTaskConfigLlms();
    return () => {
      cancelled = true;
    };
  }, [isSavedViewer]);

  useEffect(() => {
    if (isSavedViewer || !runId) return undefined;

    let cancelled = false;
    let timeoutId = null;
    let consecutiveFailures = 0;
    const baseDelayMs = trainMode ? 2500 : 5000;

    const scheduleNext = () => {
      if (cancelled) return;
      // exponential backoff up to 30s while the backend is unreachable (e.g. Render restart),
      // instead of hammering a dead instance at full rate
      const delay = consecutiveFailures > 0
        ? Math.min(baseDelayMs * 2 ** Math.min(consecutiveFailures, 4), 30000)
        : baseDelayMs;
      timeoutId = setTimeout(fetchTrainingRunStatus, delay);
    };

    const fetchTrainingRunStatus = async () => {
      try {
        const response = await apiClient.get(`/training_runs/${runId}`);
        if (cancelled) return;
        consecutiveFailures = 0;
        // safe ? access, so no need to check for undefined here
        setTrainingAblationReport(response.data?.reward_ablation || null);
        setTrainingAblationStatus(response.data?.reward_ablation_status || 'idle');
        setTrainingInsights(response.data?.training_insights || null);
        setRolloutInsights(response.data?.rollout_insights || null);
        setTrainingBehaviorReport(response.data?.training_behavior_report || null);
        setTrainingBehaviorTags(response.data?.training_behavior_tags || null);
        setRolloutBehaviorReport(response.data?.rollout_behavior_report || null);
        setRolloutBehaviorTags(response.data?.rollout_behavior_tags || null);
      } catch (error) {
        if (cancelled) return;
        consecutiveFailures += 1;
        if (consecutiveFailures <= 2 || consecutiveFailures % 5 === 0) {
          console.error(`Failed to fetch training run status (attempt ${consecutiveFailures}, backing off):`, error);
        }
      }
      scheduleNext();
    };

    fetchTrainingRunStatus();
    return () => {
      cancelled = true;
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [isSavedViewer, runId, trainMode]);

  // Basically making sure that sidebar state is alwawys up to date with the latest state in the main component, so that when users open the sidebar, they see the latest info and controls. We are adding a lot of dependencies to this useEffect, so it will run whenever any of these pieces of state change, ensuring that the sidebar always has the most current data and functions.
  useEffect(() => {
    if (!isActive) return;

    onSidebarStateChange({
      envName,
      rewardConfig,
      rewardConfigDirty,
      rewardConfigLoading,
      rewardConfigStatus,
      supportsCustomReward,
      availableRewardVariables,
      rewardFormulaExamples,
      rewardSourceLinks,
      savedRewardConfigFiles,
      selectedRewardConfigFile,
      rewardConfigSaveName,
      rewardConfigSaveSourceType,
      taskGoal,
      taskProposal,
      taskProposalLoading,
      taskProposalStatus,
      taskProposalLiveStatus,
      availableBehaviorTags,
      availableLlms,
      selectedLlmId,
      proposalStrategy,
      latestTrainingBreakdown: trainingRewardBreakdown,
      latestTrainingMeanBreakdown: trainingRewardBreakdownMean,
      latestRolloutBreakdown: rolloutRewardBreakdown,
      rewardLogs,
      onTermChange: isSavedViewer ? showSavedViewerRewardMessage : updateRewardTerm,
      onAddCustomTerm: isSavedViewer ? showSavedViewerRewardMessage : addCustomRewardTerm,
      onRemoveTerm: isSavedViewer ? showSavedViewerRewardMessage : removeRewardTerm,
      onSaveConfig: saveRewardConfig,
      onTaskGoalChange: isSavedViewer ? showSavedViewerTaskMessage : setTaskGoal,
      onLlmSelect: isSavedViewer ? showSavedViewerTaskMessage : setSelectedLlmId,
      onProposalStrategyChange: isSavedViewer ? showSavedViewerTaskMessage : setProposalStrategy,
      onProposeTaskConfig: proposeTaskConfig,
      onApplyTaskProposal: applyTaskProposal,
      onRewardConfigFileSelect: isSavedViewer ? showSavedViewerRewardMessage : setSelectedRewardConfigFile,
      onRewardConfigSaveNameChange: isSavedViewer ? showSavedViewerRewardMessage : setRewardConfigSaveName,
      onRewardConfigSaveSourceTypeChange: isSavedViewer ? showSavedViewerRewardMessage : setRewardConfigSaveSourceType,
      onSaveRewardConfigSnapshot: saveRewardConfigSnapshot,
      onLoadRewardConfigSnapshot: loadSavedRewardConfig,
    });
  }, [
    isActive,
    onSidebarStateChange,
    envName,
    rewardConfig,
    rewardConfigDirty,
    rewardConfigLoading,
    rewardConfigStatus,
    supportsCustomReward,
    availableRewardVariables,
    rewardFormulaExamples,
    rewardSourceLinks,
    savedRewardConfigFiles,
    selectedRewardConfigFile,
    rewardConfigSaveName,
    rewardConfigSaveSourceType,
    taskGoal,
    taskProposal,
    taskProposalLoading,
    taskProposalStatus,
    taskProposalLiveStatus,
    availableBehaviorTags,
    availableLlms,
    selectedLlmId,
    proposalStrategy,
    trainingRewardBreakdown,
    trainingRewardBreakdownMean,
    rolloutRewardBreakdown,
    rewardLogs,
    updateRewardTerm,
    addCustomRewardTerm,
    removeRewardTerm,
    showSavedViewerRewardMessage,
    showSavedViewerTaskMessage,
    isSavedViewer,
    saveRewardConfig,
    setSelectedLlmId,
    setProposalStrategy,
    proposeTaskConfig,
    applyTaskProposal,
    saveRewardConfigSnapshot,
    loadSavedRewardConfig,
  ]);
  /* Here we are adding envName to the dependency array of useEffect, so useEffect will rerun when envName changes*/
  useEffect(() => {
    if (isSavedViewer) return;

    let isActive = true;
    if (!runId) {
      console.warn("Run ID not set yet, cannot connect");
      return;
    }
    else {
      const connect = () => {
        const url = buildWebSocketUrl('/ws/rollout', {
          runid: runId,
          env: envName,
          train: trainMode,
          train_steps: trainSteps,
        });
        // console.log("Attempting to connect to : ", url); // DEBUG:FRONTEND
        const ws = new WebSocket(url);

        ws.onopen = () => {
          // console.log("[WebSocket] Connected ✅"); // DEBUG:FRONTEND
          socketRef.current = ws;
          retryRef.current = null;
        };

        ws.onmessage = (event) => {
          //console.log("Is Active is ", isActive, "event is currently ", event);
          if (!isActive) return;
          /* DO NOT UPDATE THE STATE IF THE SIMULATION IS PAUSED*/
          if (isPausedRef.current) return;

          const data = JSON.parse(event.data);
          console.log("Received data type ", data.type);
          if (data.type === "session"){
            setSessionId(data.session_id);
            // the websocket does not need to record any more data
            return; 
          } else if (data.type === "tick") {
            // set the training rollouts to the right value
            console.log("Received tick data: ", data);
            const rewardData = {
              step: data.step,
              evalReward: data.eval_reward ?? null,
              rewardMean: data.reward_mean ?? null,
              reward: data.reward ?? null,
              breakdown: data.reward_breakdown || {},
              breakdownMean: data.reward_breakdown_mean || {},
            };
            setTrainingRollouts((prev) => [rewardData, ...prev]);
            setTrainingRewardBreakdown(data.reward_breakdown || {});
            setTrainingRewardBreakdownMean(data.reward_breakdown_mean || {});
            if (Array.isArray(data.new_training_episodes) && data.new_training_episodes.length > 0) {
              setTrainingEpisodes((prev) => [...data.new_training_episodes.slice().reverse(), ...prev]);
              setSelectedTrainingTimelineEpisode(data.new_training_episodes[data.new_training_episodes.length - 1]?.episode ?? null);
            }
            appendRewardLog({
              source: 'training',
              label: `Step ${data.step}`,
              total: data.reward_breakdown?.total ?? data.reward ?? 0,
              breakdown: data.reward_breakdown || {},
              at: data.ts ? new Date(data.ts * 1000).toLocaleTimeString() : 'training update',
            });
          } else {
            console.log("Received Episode data: ", data.type, data);
            // set the episode info to the value of data.episode
            const rawEpisodeNumber = Number(data.episode ?? 0);
            if (rolloutEpisodeOffsetRef.current === null || rolloutEpisodeOffsetRef.current === undefined) {
              rolloutEpisodeOffsetRef.current = rawEpisodeNumber;
            }
            const rolloutEpisodeNumber = rawEpisodeNumber - rolloutEpisodeOffsetRef.current;
            const rawSimFrameEpisodeNumber = data.sim_frame_episode_number;
            const simFrameEpisodeNumber =
              rawSimFrameEpisodeNumber !== null && rawSimFrameEpisodeNumber !== undefined
                ? Number(rawSimFrameEpisodeNumber) - rolloutEpisodeOffsetRef.current
                : null;
            // if data.type is not session
            if(data.ep_frames.length > 0){
              setFrames(data.ep_frames);        // store all frames
              if (simFrameEpisodeNumber !== null && simFrameEpisodeNumber !== undefined) {
                setCapturedEpisodeFramesByEpisode((prev) => ({
                  ...prev,
                  [simFrameEpisodeNumber]: data.ep_frames,
                }));
                setSelectedVisualizationEpisode((prev) =>
                  prev === null || prev === undefined ? simFrameEpisodeNumber : prev
                );
              }
            }
            // console.log("Episode: ", data.episode, "   Reward: ", data.reward); // DEBUG:FRONTEND
            // console.log("Frames received length: ", data.ep_frames.length); // DEBUG:FRONTEND
            // console.log("Data sim frame episode number: ", data.sim_frame_episode_number); // DEBUG:FRONTEND
            if(simFrameEpisodeNumber !== null && simFrameEpisodeNumber !== undefined) {
              setEpisodeNumForSimulation(simFrameEpisodeNumber);
            }
            //setIsPlaying(true); <- playback controlled by isPlaying var           // start playback automatically
            // don't need all the other information
            const newData = {
              reward: data.reward,
              episode: rolloutEpisodeNumber,
              reward_breakdown: data.reward_breakdown || {},
              reward_raw_terms: data.reward_raw_terms || {},
              reward_history: data.reward_history || [],
              episode_outcome: data.episode_outcome || 'unknown',
              episode_outcome_reason: data.episode_outcome_reason || 'outcome unavailable',
              episode_terminal_timestep: data.episode_terminal_timestep ?? null,
              terminated: Boolean(data.terminated),
              truncated: Boolean(data.truncated),
            };
            if (!trainMode) {
              setEpisodeInfo({ episode: rolloutEpisodeNumber, reward: data.reward });
              setRolloutRewardBreakdown(data.reward_breakdown || {});
              setLatestRolloutRawTerms(data.reward_raw_terms || {});
              setRollouts((prev) => [newData, ...prev]);
              appendRewardLog({
                source: 'rollout',
                label: `Episode ${rolloutEpisodeNumber}`,
                total: data.reward,
                breakdown: data.reward_breakdown || {},
                at: new Date().toLocaleTimeString(),
              });
            }
          };
        };
        ws.onerror = (err) => console.error("WebSocket Error: ", err);
        ws.onclose = () => {
          // console.log("[WebSocket] Disconnected ❌"); // DEBUG:FRONTEND
          // console.log("WebSocket is Active: ", isActive); // DEBUG:FRONTEND
          if(!isActive) return;
          
          // console.log("WebSocket Disconnected, retrying in 1s. "); // DEBUG:FRONTEND
          retryRef.current = setTimeout(connect, 1000); // retry after 1 second
        }
      }
      
      setRollouts([]); // restart the graph simulation from the beginning, upon new simulation
      setTrainingRollouts([]);
      setTrainingEpisodes([]);
      setTrainingRewardBreakdown({});
      setTrainingRewardBreakdownMean({});
      setRolloutRewardBreakdown({});
      setRewardLogs([]);
      setCapturedEpisodeFramesByEpisode({});
      setSelectedVisualizationEpisode(null);
      setSelectedTimelineEpisode(null);
      setSelectedTimelineStep(null);
      // initial attempt
      togglePause(false);
      connect();
      // console.log("envName: ", envName); // DEBUG:FRONTEND
      // console.log("trainMode: ", trainMode);// DEBUG:FRONTEND
      

      // console.log("frames: ", frames &&frames.length) // DEBUG:FRONTEND
      return () => {
        // cleanup function, run before next component runs
        isActive = false;
        if (socketRef.current) socketRef.current.close();
        if (retryRef.current) clearTimeout(retryRef.current);
      };
    }
  }, [envName, isSavedViewer, rolloutSessionVersion, trainMode, runId]);

  // This is the useEffect for the frame Data from the video
  useEffect(() => {
    if (!isPlaying || (visualizationFrames && visualizationFrames.length) === 0) return;
    //advance frame at ferquency of 20fps
    intervalRef.current = setInterval(() => {
      setCurrentFrame((prev) => {
        if (Array.isArray(visualizationFrames) && prev < visualizationFrames.length - 1) return prev + 1;  // advance frame
        // want to implement looping, so no stop at end
        //clearInterval(intervalRef.current);             // stop at end
        return 0;
      });
    }, replayInterval); // ~20 FPS

    return () => clearInterval(intervalRef.current); // clean up
  }, [isPlaying, replayInterval, visualizationFrames]);

  const handlePlay = () => setIsPlaying(true);
  const handlePause = () => {
    // console.log("handlePause"); // DEBUG:FRONTEND
    setIsPlaying(false);
    // console.log("isPlaying.current: ", isPlaying); // DEBUG:FRONTEND
    clearInterval(intervalRef.current);
  };
  const handleRestart = () => {
    setCurrentFrame(0);
    setIsPlaying(true);
  };

  const toggleTrainMode = async () => {
    // safety check to prevent multiple rapid clicks causing issues
    if (trainMode) return;
    openPathPopup();
    // process the trainMode variable WITHIN the popup (i.e. after popup closes)
  }

  const stopTraining = async () => {
    if (!runId || stoppingTraining) return;
    setStoppingTraining(true);
    try {
      await apiClient.post("/stop_training", { run_id: runId });
      setTrainMode(false);
    } catch (e) {
      console.error("Failed to stop training:", e);
    } finally {
      setStoppingTraining(false);
    }
  };
  const buttonStyle = (bg) => ({
    padding: '0.6rem 1rem',
    background: bg,
    color: 'white',
    border: '1px solid rgba(255,255,255,0.18)',
    borderRadius: '14px',
    fontWeight: 700,
    fontSize: '0.96rem',
    cursor: 'pointer',
    boxShadow: '0 14px 28px rgba(15, 23, 42, 0.12)',
    transition: 'all 0.2s ease',
  });

  const handleModelUpload = (e) => {
    const f = e.target.files?.[0] || null;
    setFile(f);
  };
  const applyModelSelection = useCallback(async (modelName) => {
    if (!runId) {
      console.warn("Run ID not set yet, cannot load model");
      return;
    }
    setLoading(true);
    try {
      const response = await apiClient.post("/load_model", { run_id: runId, model_name: modelName, env_name: envName });
      if (!response?.data?.ok) {
        const errorMessage =
          response?.data?.error === 'model_env_mismatch'
            ? `Model is for ${response?.data?.model_env_name || 'another environment'}, but the current environment is ${response?.data?.expected_env_name || envName}.`
            : response?.data?.error || "Failed to load model.";
        setRewardConfigStatus(`Model load failed: ${errorMessage}`);
        return;
      }
      const loadedModelName = response?.data?.model || "";
      setIsUsingNone(!loadedModelName);
      setActiveModelName(loadedModelName);
      setRewardConfigStatus(
        loadedModelName
          ? `Using model ${loadedModelName} for rollout.`
          : 'Using no model. Rollout is running with the random policy.'
      );
      restartLiveRolloutSession();
    } finally {
      setLoading(false);
    }
  }, [envName, restartLiveRolloutSession, runId]);

  const openModelSwitchSavePrompt = useCallback((modelName) => {
    setPendingModelSwitch({ modelName });
    setSavePopupMode('model-switch');
    setShowSavePopup(true);
  }, []);

  const cancelPendingModelSwitch = useCallback(() => {
    setPendingModelSwitch(null);
    setSavePopupMode('manual');
    setShowSavePopup(false);
  }, []);

  const continueModelSwitchWithoutSaving = useCallback(async () => {
    if (!pendingModelSwitch) return;
    const modelName = pendingModelSwitch.modelName;
    setPendingModelSwitch(null);
    setSavePopupMode('manual');
    setShowSavePopup(false);
    await applyModelSelection(modelName);
  }, [applyModelSelection, pendingModelSwitch]);

  const useNone = async () => {
    setLoading(true);
    try {
      // “Clear” the session’s model by loading none; implement either:
      // 1) a dedicated endpoint:
      // await apiClient.post("/unload_model", { session_id: sessionId });
      // OR 2) overload load_model with a sentinel:
      const response = await apiClient.post("/load_model", { run_id: runId, model_name: "", env_name: envName });
      if (!response?.data?.ok) {
        const errorMessage = response?.data?.error || "Failed to clear model.";
        setRewardConfigStatus(`Model clear failed: ${errorMessage}`);
        return;
      }
      setIsUsingNone(true);
      setActiveModelName("");
      setRewardConfigStatus('Using no model. Rollout is running with the random policy.');
      restartLiveRolloutSession();
    } finally {
      setLoading(false);
    }
  };

  const loadServerModel = async () => {
    if (!runId) {
      console.warn("Run ID not set yet, cannot load model");
      return;
    }
    const nextModelName = selectedServerModel || "";
    if (rollouts.length > 0) {
      openModelSwitchSavePrompt(nextModelName);
      return;
    }
    await applyModelSelection(nextModelName);
  };

  const handleSave = async (filename) => {
    if (!rollouts || rollouts.length === 0){
      alert("No rollouts to save.");
      return;
    }
    if (!filename) {
      alert("Please enter a filename! filename is empty.");
      return;
    }
    setSavingRollouts(true);
    try {
      const res = await apiClient.post("/save_rollouts_data", { run_id: runId, rollout_filename: filename, rollouts: rollouts});
      const filesRes = await apiClient.get("/rollouts_files");
      const files = filesRes.data.rollouts || [];
      setRolloutFiles(files);
      setSelectedRolloutFile(filename && files.includes(filename) ? filename : files[0] || "");
      // console.log("Saved rollout data with status:  ", res.status); // DEBUG:FRONTEND
    } catch (err) {
      console.error("Failed to save: ", err);
      alert("Failed to save rollout data: " + err);
    } finally {
      setSavingRollouts(false);
      setShowSavePopup(false);
      const pendingModelName = pendingModelSwitch?.modelName;
      setPendingModelSwitch(null);
      setSavePopupMode('manual');
      if (pendingModelName !== undefined && pendingModelName !== null) {
        await applyModelSelection(pendingModelName);
      }
    }
  };

  const loadSavedRollouts = async () => {
    if (!selectedRolloutFile) {
      alert("No rollout file selected.");
      return;
    }
    setLoadingSavedRollouts(true);
    try {
      const res = await apiClient.post("/load_rollouts_data", {
        rollout_filename: selectedRolloutFile,
      });
      const loadedRollouts = Array.isArray(res.data?.rollouts) ? res.data.rollouts : [];
      onOpenLoadedRollout({
        rollouts: loadedRollouts,
        envName,
        fileName: selectedRolloutFile,
      });
    } catch (err) {
      console.error("Failed to load rollouts:", err);
      alert("Failed to load rollout data: " + err);
    } finally {
      setLoadingSavedRollouts(false);
    }
  };

  const uploadAndLoad = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file); // field name "file" expected by backend
      const up = await apiClient.post("/upload_model", form);
      const modelName = up.data?.model_name; // backend should return stored filename
      if (modelName) {
        setSelectedServerModel(modelName);
        await applyModelSelection(modelName);
      }
    } finally {
      setLoading(false);
      setFile(null);
    }
  };

  const getRootSavedModelsLink = async() => {
    const result = await apiClient.get("/get_model_path");
    // console.log("Root saved models link from backend:", result.data); // DEBUG:FRONTEND
    let path = result.data;
    try {
      await navigator.clipboard.writeText(path);
      setFilePathCopied(true);
      setTimeout(() => setFilePathCopied(false), 1500); // reset after 1.5s
    } catch (err) {
      console.error("Failed to copy: ", err);
    }
  };
  const workspaceShellStyle = {
    background: 'linear-gradient(180deg, rgba(255,255,255,0.84), rgba(241,245,249,0.86))',
    borderRadius: '28px',
    border: '1px solid rgba(148, 163, 184, 0.16)',
    boxShadow: '0 26px 54px rgba(15, 23, 42, 0.08)',
    backdropFilter: 'blur(18px)',
    padding: '1.45rem',
  };

  const workspaceHeaderStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: '1rem',
    flexWrap: 'wrap',
    marginBottom: '1rem',
  };

  const workspaceMetaStyle = {
    color: '#526277',
    fontSize: '0.94rem',
    maxWidth: '52rem',
    lineHeight: 1.65,
  };

  const sectionPanelStyle = {
    background: 'linear-gradient(180deg, rgba(255,255,255,0.84), rgba(248,250,252,0.72))',
    borderRadius: '20px',
    border: '1px solid rgba(148, 163, 184, 0.16)',
    padding: '1.05rem',
    marginTop: '1rem',
    boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.55), 0 10px 24px rgba(15, 23, 42, 0.05)',
  };

  const statusBadgeStyle = (backgroundColor) => ({
    padding: '0.55rem 0.95rem',
    borderRadius: '999px',
    backgroundColor,
    color: '#fff',
    fontWeight: 700,
    boxShadow: '0 8px 20px rgba(15, 23, 42, 0.12)',
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.45rem',
  });
  const renderInsightCards = (report, emptyLabel) => {
    const summary = report?.summary || {};
    const cards = Array.isArray(report?.insights) ? report.insights : [];
    if (cards.length === 0) {
      return (
        <div style={{ color: '#64748b', fontSize: '0.9rem' }}>
          {emptyLabel}
        </div>
      );
    }
    return (
      <div style={{ display: 'grid', gap: '0.75rem' }}>
        <div style={{ color: '#64748b', fontSize: '0.82rem' }}>
          Analyzed {summary.episodes_analyzed || 0} episodes with {summary.success_count || 0} successes and {summary.failure_count || 0} failures.
          {summary.success_rate !== null && summary.success_rate !== undefined ? ` Success rate: ${(summary.success_rate * 100).toFixed(0)}%.` : ''}
        </div>
        {cards.map((insight, index) => (
          <div
            key={`${insight.category || 'insight'}-${index}`}
            style={{
              border: '1px solid rgba(148, 163, 184, 0.22)',
              borderRadius: '12px',
              padding: '0.85rem 0.9rem',
              backgroundColor: 'rgba(255,255,255,0.62)',
              textAlign: 'left',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
              <div style={{ fontWeight: 800, color: '#334155' }}>{insight.title}</div>
              <div style={{ fontSize: '0.76rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                {insight.priority || 'info'} • confidence {Math.round((insight.confidence || 0) * 100)}%
              </div>
            </div>
            <div style={{ color: '#475569', fontSize: '0.88rem', marginTop: '0.35rem', lineHeight: 1.5 }}>
              {insight.body}
            </div>
            {insight.evidence && (
              <div style={{ marginTop: '0.45rem', fontSize: '0.76rem', color: '#64748b', fontFamily: 'ui-monospace, SFMono-Regular, monospace' }}>
                {Object.entries(insight.evidence).map(([key, value]) => `${key}: ${value}`).join(' | ')}
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };
  const renderBehaviorPlanCards = (proposal) => {
    const plan = proposal?.behavior_plan;
    const availableTags = Array.isArray(proposal?.available_behavior_tags) ? proposal.available_behavior_tags : availableBehaviorTags;
    if (!plan && (!availableTags || availableTags.length === 0)) {
      return null;
    }

    const renderPlanList = (items, emptyLabel, accentColor) => (
      Array.isArray(items) && items.length > 0 ? (
        <div style={{ display: 'grid', gap: '0.45rem', marginTop: '0.45rem' }}>
          {items.map((item) => (
            <div
              key={`${accentColor}-${item.key}`}
              style={{
                border: '1px solid rgba(148, 163, 184, 0.18)',
                borderRadius: '10px',
                padding: '0.65rem 0.75rem',
                backgroundColor: 'rgba(255,255,255,0.52)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
                <div style={{ fontWeight: 700, color: '#334155' }}>{item.key}</div>
                <div style={{ color: accentColor, fontFamily: 'ui-monospace, SFMono-Regular, monospace', fontSize: '0.8rem' }}>
                  weight {Number(item.weight || 0).toFixed(2)}
                </div>
              </div>
              {item.reason && (
                <div style={{ color: '#64748b', fontSize: '0.8rem', marginTop: '0.25rem', lineHeight: 1.45 }}>
                  {item.reason}
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div style={{ color: '#64748b', fontSize: '0.82rem', marginTop: '0.45rem' }}>{emptyLabel}</div>
      )
    );

    return (
      <div style={{ display: 'grid', gap: '0.85rem', marginTop: '0.95rem' }}>
        <div
          style={{
            border: '1px solid rgba(148, 163, 184, 0.22)',
            borderRadius: '14px',
            padding: '0.9rem',
            backgroundColor: 'rgba(255,255,255,0.58)',
            textAlign: 'left',
          }}
        >
          <div style={{ fontWeight: 800, color: '#334155' }}>Behavior Plan</div>
          {plan?.rationale && (
            <div style={{ color: '#64748b', fontSize: '0.82rem', marginTop: '0.3rem', lineHeight: 1.5 }}>
              {plan.rationale}
            </div>
          )}
          <div style={{ marginTop: '0.7rem' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#0f766e' }}>Desired Behaviors</div>
            {renderPlanList(plan?.desired_tags, 'No desired behavior tags selected.', '#0f766e')}
          </div>
          <div style={{ marginTop: '0.8rem' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#b45309' }}>Avoid Behaviors</div>
            {renderPlanList(plan?.avoid_tags, 'No avoid behavior tags selected.', '#b45309')}
          </div>
        </div>
        {Array.isArray(availableTags) && availableTags.length > 0 && (
          <div
            style={{
              border: '1px solid rgba(148, 163, 184, 0.18)',
              borderRadius: '14px',
              padding: '0.9rem',
              backgroundColor: 'rgba(255,255,255,0.48)',
              textAlign: 'left',
            }}
          >
            <div style={{ fontWeight: 800, color: '#334155' }}>Available Behavior Tags</div>
            <div style={{ color: '#64748b', fontSize: '0.8rem', marginTop: '0.25rem' }}>
              {availableTags.length} tags available for the current environment.
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.45rem', marginTop: '0.65rem' }}>
              {availableTags.map((tag) => (
                <span
                  key={`available-behavior-tag-${tag.key}`}
                  style={{
                    padding: '0.32rem 0.6rem',
                    borderRadius: '999px',
                    backgroundColor: tag.polarity === 'avoid' ? 'rgba(251, 191, 36, 0.16)' : 'rgba(14, 165, 233, 0.12)',
                    border: `1px solid ${tag.polarity === 'avoid' ? 'rgba(217, 119, 6, 0.22)' : 'rgba(14, 165, 233, 0.2)'}`,
                    color: '#334155',
                    fontSize: '0.76rem',
                    fontWeight: 700,
                  }}
                >
                  {tag.key}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };
  const renderBehaviorReportCards = (report, tagReport, emptyLabel) => {
    const metrics = report?.metrics || {};
    const notableMetrics = Array.isArray(report?.notable_metrics) ? report.notable_metrics : [];
    const summary = report?.summary || {};
    const supportedTags = Array.isArray(tagReport?.supported_tags) ? tagReport.supported_tags : [];
    const tentativeTags = Array.isArray(tagReport?.tentative_tags) ? tagReport.tentative_tags : [];

    if (Object.keys(metrics).length === 0 && supportedTags.length === 0 && tentativeTags.length === 0) {
      return <div style={{ color: '#64748b', fontSize: '0.9rem' }}>{emptyLabel}</div>;
    }

    const renderTagPills = (items, accentColor, title) => (
      <div style={{ marginTop: '0.7rem' }}>
        <div style={{ fontSize: '0.82rem', fontWeight: 700, color: accentColor }}>{title}</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.45rem', marginTop: '0.45rem' }}>
          {items.length === 0 ? (
            <span style={{ color: '#64748b', fontSize: '0.8rem' }}>None yet.</span>
          ) : (
            items.map((tag) => (
              <span
                key={`${title}-${tag.key}`}
                style={{
                  padding: '0.32rem 0.6rem',
                  borderRadius: '999px',
                  backgroundColor: 'rgba(255,255,255,0.7)',
                  border: '1px solid rgba(148, 163, 184, 0.22)',
                  color: '#334155',
                  fontSize: '0.76rem',
                  fontWeight: 700,
                }}
              >
                {tag.key} ({Math.round((Number(tag.score || 0)) * 100)}%)
              </span>
            ))
          )}
        </div>
      </div>
    );

    return (
      <div style={{ display: 'grid', gap: '0.8rem', marginTop: '0.95rem' }}>
        <div
          style={{
            border: '1px solid rgba(148, 163, 184, 0.22)',
            borderRadius: '14px',
            padding: '0.9rem',
            backgroundColor: 'rgba(255,255,255,0.58)',
            textAlign: 'left',
          }}
        >
          <div style={{ fontWeight: 800, color: '#334155' }}>Behavior Summary</div>
          <div style={{ color: '#64748b', fontSize: '0.82rem', marginTop: '0.3rem', lineHeight: 1.5 }}>
            Episodes analyzed: {summary.episodes_analyzed || 0}
            {summary.primary_label ? ` · dominant label: ${summary.primary_label}` : ''}
          </div>
          {renderTagPills(supportedTags, '#0f766e', 'Supported Behavior Tags')}
          {renderTagPills(tentativeTags, '#475569', 'Tentative Behavior Tags')}
        </div>
        <div
          style={{
            border: '1px solid rgba(148, 163, 184, 0.18)',
            borderRadius: '14px',
            padding: '0.9rem',
            backgroundColor: 'rgba(255,255,255,0.48)',
            textAlign: 'left',
          }}
        >
          <div style={{ fontWeight: 800, color: '#334155' }}>Notable Deterministic Metrics</div>
          <div style={{ display: 'grid', gap: '0.45rem', marginTop: '0.6rem' }}>
            {(notableMetrics.length > 0 ? notableMetrics : Object.entries(metrics).slice(0, 10).map(([name, value]) => ({ name, value }))).map((metric, index) => (
              <div
                key={`${metric.name || 'metric'}-${index}`}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  gap: '1rem',
                  padding: '0.45rem 0',
                  borderBottom: '1px solid rgba(148, 163, 184, 0.12)',
                  fontSize: '0.8rem',
                }}
              >
                <span style={{ color: '#475569' }}>{metric.name}</span>
                <span style={{ color: '#0f172a', fontFamily: 'ui-monospace, SFMono-Regular, monospace' }}>
                  {Number(metric.value || 0).toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };
  return (
  <div
    style={{
      padding: '1.25rem',
      fontFamily: 'Segoe UI, sans-serif',
      width: '100%',
      maxWidth: '1040px',
      margin: '0 auto',
      display: 'grid',
      gap: '1.25rem',
    }}
  >
    {isSavedViewer && (
      <div
        style={{
          marginBottom: '1rem',
          padding: '0.75rem 1rem',
          background: 'rgba(71, 85, 105, 0.12)',
          border: '1px solid rgba(71, 85, 105, 0.25)',
          borderRadius: '10px',
          color: '#334155',
          fontWeight: 600,
          textAlign: 'center',
        }}
      >
        Saved Rollout Viewer{viewerLabel ? `: ${viewerLabel}` : ''}
      </div>
    )}
    <div style={{ ...workspaceShellStyle, textAlign: 'center', padding: '1.25rem 1.5rem' }}>
    <h1 style={{ fontSize: '2.2rem', fontWeight: 700, textAlign: 'center', color: '#4f46e5', margin: 0 }}>
      ⚡ OpenGym Copilot
    </h1>
    <div style={{ marginTop: '0.35rem', color: '#64748b', fontSize: '0.95rem' }}>
      Training and rollout playback now live in separate workspaces so they read as two different jobs.
    </div>
    </div>

    <div style={workspaceShellStyle}>
    <div style={workspaceHeaderStyle}>
      <div style={{ textAlign: 'left' }}>
        <h3 style={{ fontSize: '1.6rem', margin: 0, color: '#3b82f6' }}>🧠 Training Workspace</h3>
        <div style={workspaceMetaStyle}>
          Configure the environment, total steps, and training session here. This area is only about background learning and policy progress.
        </div>
      </div>
      <div style={statusBadgeStyle(isSavedViewer ? '#94a3b8' : trainMode ? '#3b82f6' : '#94a3b8')}>
        <span>{trainMode ? 'Training Active' : 'Training Idle'}</span>
      </div>
    </div>
      
    {/* Top Control Row */}
    <div
      style={{
        ...sectionPanelStyle,
        display: 'flex',
        gap: '1rem',
        alignItems: 'center',
        justifyContent: 'center',
        flexWrap: 'wrap',
      }}
    >
      <button
        onClick={trainMode ? stopTraining : toggleTrainMode}
        disabled={isSavedViewer || stoppingTraining}
        style={{
          padding: '0.7rem 1.25rem',
          fontSize: '1rem',
          backgroundColor: isSavedViewer ? '#cbd5e1' : stoppingTraining ? '#94a3b8' : trainMode ? '#dc2626' : '#0f766e',
          color: 'white',
          border: 'none',
          borderRadius: '999px', // pill shape
          cursor: isSavedViewer || stoppingTraining ? 'not-allowed' : 'pointer',
          boxShadow: trainMode ? '0 8px 18px rgba(220, 38, 38, 0.22)' : '0 8px 18px rgba(15, 118, 110, 0.18)',
          transition: 'all 0.3s ease-in-out',
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.5rem',
          fontWeight: 700,
          opacity: isSavedViewer ? 0.7 : 1,
        }}
      >
        {stoppingTraining ? 'Stopping...' : trainMode ? '■ Stop Training' : '▶ Start Training'}
      </button>
      <SetPathPopup
        isOpen={showPathPopup}
        defaultPath={frozenPath !== null ? frozenPath : `ppo_model_${envName}_${timestamp}.zip`}
        defaultTrainSteps={trainSteps}
        defaultHyperparams={trainingHyperparams}
        onConfirm={saveTrainingPath}
        onClose={closePathPopup}
      />
      <SaveRolloutPopup
        runId={runId}
        isOpen={showSavePopup}
        onConfirm={handleSave}
        onClose={savePopupMode === 'model-switch' ? cancelPendingModelSwitch : closeShowSavePopup}
        onSkip={savePopupMode === 'model-switch' ? continueModelSwitchWithoutSaving : null}
        showSkip={savePopupMode === 'model-switch'}
        title={savePopupMode === 'model-switch' ? 'Save Current Rollout Before Model Switch' : 'Save Rollouts'}
        message={savePopupMode === 'model-switch'
          ? 'You are about to start a fresh rollout run with the selected model. Save the current rollout history first, or continue without saving.'
          : ''}
        confirmLabel={savePopupMode === 'model-switch' ? 'Save And Switch Model' : 'Save'}
        skipLabel="Switch Without Saving"
      />

      <label htmlFor="envSelect" style={{ fontWeight: 600 }}>Environment:</label>
      <select
        id="envSelect"
        value={envName}
        onChange={handleEnvChange}
        disabled={isSavedViewer}
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
        <option value="CartPoleLoose-v0">CartPoleLoose-v0</option>
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
      <div
        style={{
          ...sectionPanelStyle,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '0.75rem',
          flexWrap: 'wrap',
        }}
      >
        <button
          onClick={() => setTrainingWorkspaceViewIndex((prev) => (prev - 1 + trainingWorkspaceViews.length) % trainingWorkspaceViews.length)}
          style={{
            width: '36px',
            height: '36px',
            borderRadius: '999px',
            border: '1px solid #cbd5e1',
            backgroundColor: 'white',
            color: '#334155',
            fontWeight: 700,
          }}
          aria-label="Show previous training panel"
        >
          {'<'}
        </button>
        <div style={{ textAlign: 'center', flex: '1 1 240px' }}>
          <div style={{ fontSize: '1.05rem', fontWeight: 800, color: '#334155' }}>{currentTrainingWorkspaceView.title}</div>
          <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
            {trainingWorkspaceViewIndex + 1} / {trainingWorkspaceViews.length}
          </div>
        </div>
        <select
          value={String(trainingWorkspaceViewIndex)}
          onChange={(event) => setTrainingWorkspaceViewIndex(Number(event.target.value))}
          style={{
            minWidth: '240px',
            padding: '0.45rem 0.6rem',
            borderRadius: '8px',
            border: '1px solid #cbd5e1',
            backgroundColor: 'white',
            color: '#334155',
            fontSize: '0.92rem',
          }}
          aria-label="Choose training workspace view"
        >
          {trainingWorkspaceViews.map((view, index) => (
            <option key={view.key} value={index}>
              {view.title}
            </option>
          ))}
        </select>
        {!isSavedViewer && (
          <button
            onClick={() => setShowTrainingInsights((prev) => !prev)}
            style={{
              padding: '0.5rem 0.9rem',
              borderRadius: '999px',
              border: '1px solid rgba(15, 118, 110, 0.22)',
              backgroundColor: showTrainingInsights ? '#0f766e' : 'white',
              color: showTrainingInsights ? 'white' : '#0f766e',
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            {showTrainingInsights ? 'Hide Insights' : 'Show Insights'}
          </button>
        )}
        <button
          onClick={() => setTrainingWorkspaceViewIndex((prev) => (prev + 1) % trainingWorkspaceViews.length)}
          style={{
            width: '36px',
            height: '36px',
            borderRadius: '999px',
            border: '1px solid #cbd5e1',
            backgroundColor: 'white',
            color: '#334155',
            fontWeight: 700,
          }}
          aria-label="Show next training panel"
        >
          {'>'}
        </button>
      </div>
    )}
    {trainMode && currentTrainingWorkspaceView.key === 'reward_chart' && (
      <div
        style={{
          ...sectionPanelStyle,
          width: '100%',
          maxWidth: '100%',
          minWidth: 0,
          margin: '0 auto',
          paddingBottom: '0.5rem',
          overflow: 'hidden',
          boxSizing: 'border-box',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '0.75rem',
            marginBottom: '0.85rem',
            flexWrap: 'wrap',
          }}
        >
          <button
            onClick={() => setTrainingGraphIndex((prev) => (prev - 1 + trainingGraphDefinitions.length) % trainingGraphDefinitions.length)}
            disabled={trainingGraphDefinitions.length <= 1}
            style={{
              width: '32px',
              height: '32px',
              borderRadius: '999px',
              border: '1px solid #cbd5e1',
              backgroundColor: 'white',
              cursor: trainingGraphDefinitions.length <= 1 ? 'not-allowed' : 'pointer',
              fontWeight: 700,
              color: '#334155',
            }}
            aria-label="Show previous training graph"
          >
            {'<'}
          </button>
          <div style={{ textAlign: 'center', flex: '1 1 240px' }}>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#334155' }}>{currentTrainingGraph.title}</div>
            <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
              {trainingGraphDefinitions.length > 0 ? `${trainingGraphIndex + 1} / ${trainingGraphDefinitions.length}` : 'No training data yet'}
            </div>
          </div>
          <select
            value={String(trainingGraphIndex)}
            onChange={(e) => setTrainingGraphIndex(Number(e.target.value))}
            style={{
              minWidth: '240px',
              padding: '0.45rem 0.6rem',
              borderRadius: '8px',
              border: '1px solid #cbd5e1',
              backgroundColor: 'white',
              color: '#334155',
              fontSize: '0.92rem',
            }}
            aria-label="Choose training reward term graph"
          >
            {trainingGraphDefinitions.map((graph, index) => (
              <option key={graph.title} value={index}>
                {graph.title}
              </option>
            ))}
          </select>
          <button
            onClick={() => setTrainingGraphIndex((prev) => (prev + 1) % trainingGraphDefinitions.length)}
            disabled={trainingGraphDefinitions.length <= 1}
            style={{
              width: '32px',
              height: '32px',
              borderRadius: '999px',
              border: '1px solid #cbd5e1',
              backgroundColor: 'white',
              cursor: trainingGraphDefinitions.length <= 1 ? 'not-allowed' : 'pointer',
              fontWeight: 700,
              color: '#334155',
            }}
            aria-label="Show next training graph"
          >
            {'>'}
          </button>
        </div>
        <div
          style={{
            position: 'relative',
            width: '100%',
            maxWidth: '100%',
            minWidth: 0,
            height: '300px',
            overflow: 'hidden',
          }}
        >
          <Line
            data={{
              labels: currentTrainingGraph.labels,
              datasets: currentTrainingGraph.datasets.map((dataset) => ({
                ...dataset,
                fill: false,
                tension: 0.25,
                pointRadius: 0,
                pointHoverRadius: 3,
                borderWidth: 2,
              })),
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              animation: false,
              normalized: true,
              interaction: {
                intersect: false,
                mode: 'index',
              },
              scales: {
                x: {
                  title: { display: true, text: "Training Steps" },
                  ticks: {
                    autoSkip: true,
                    maxTicksLimit: 10,
                  },
                },
                y: { title: { display: true, text: "Reward" } },
              },
            }}
          />
        </div>
      </div>
    )}
    {trainMode && currentTrainingWorkspaceView.key === 'temporal_breakdown' && (
      <div style={{ ...sectionPanelStyle, marginTop: '1rem' }}>
        <h3 style={{ fontSize: '1.2rem', color: '#0f766e', marginTop: 0 }}>Training Episode Temporal Breakdown</h3>
        <div style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: '0.8rem' }}>
          Inspect one completed training episode timestep-by-timestep to see which reward term pushed the policy toward failure or success.
        </div>
        <div
          style={{
            display: 'flex',
            gap: '0.75rem',
            alignItems: 'center',
            justifyContent: 'center',
            flexWrap: 'wrap',
            marginBottom: '0.9rem',
          }}
        >
          <label htmlFor="trainingTimelineMode" style={{ fontWeight: 600 }}>Mode</label>
          <select
            id="trainingTimelineMode"
            value={trainingTimelineMode}
            onChange={(event) => setTrainingTimelineMode(event.target.value)}
            style={{ padding: '0.4rem 0.55rem', minWidth: 180 }}
          >
            <option value="episode">Single Episode</option>
            <option value="average">Average By Type</option>
          </select>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
            <span style={{ fontWeight: 600 }}>Episode Type</span>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
              <input
                type="radio"
                name="trainingTimelineOutcome"
                value="all"
                checked={trainingTimelineOutcomeFilter === 'all'}
                onChange={(event) => setTrainingTimelineOutcomeFilter(event.target.value)}
              />
              All
            </label>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
              <input
                type="radio"
                name="trainingTimelineOutcome"
                value="success"
                checked={trainingTimelineOutcomeFilter === 'success'}
                onChange={(event) => setTrainingTimelineOutcomeFilter(event.target.value)}
              />
              Successful ({trainingOutcomeCounts.success})
            </label>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
              <input
                type="radio"
                name="trainingTimelineOutcome"
                value="failure"
                checked={trainingTimelineOutcomeFilter === 'failure'}
                onChange={(event) => setTrainingTimelineOutcomeFilter(event.target.value)}
              />
              Failed ({trainingOutcomeCounts.failure})
            </label>
          </div>
          <label htmlFor="trainingTimelineEpisode" style={{ fontWeight: 600 }}>Episode</label>
          <select
            id="trainingTimelineEpisode"
            value={selectedTrainingTimelineEpisode ?? ''}
            onChange={(event) => setSelectedTrainingTimelineEpisode(Number(event.target.value))}
            style={{ padding: '0.4rem 0.55rem', minWidth: 160 }}
            disabled={trainingTimelineMode === 'average' || orderedFilteredTrainingEpisodes.length === 0}
          >
            {orderedFilteredTrainingEpisodes.length === 0 ? (
              <option value="">No training episodes yet</option>
            ) : (
              orderedFilteredTrainingEpisodes.map((entry) => (
                <option key={`training-episode-${entry.episode}`} value={entry.episode}>
                  Episode {entry.episode}
                </option>
              ))
            )}
          </select>
          <label htmlFor="trainingTimelineView" style={{ fontWeight: 600 }}>View</label>
          <select
            id="trainingTimelineView"
            value={String(trainingTimelineGraphIndex)}
            onChange={(event) => setTrainingTimelineGraphIndex(Number(event.target.value))}
            style={{ padding: '0.4rem 0.55rem', minWidth: 300 }}
            disabled={trainingTimelineGraphDefinitions.length === 0}
          >
            {trainingTimelineGraphDefinitions.length === 0 ? (
              <option value="0">No training reward history yet</option>
            ) : (
              trainingTimelineGraphDefinitions.map((graph, index) => (
                <option key={graph.title} value={index}>
                  {graph.title}
                </option>
              ))
            )}
          </select>
        </div>
        <div style={{ textAlign: 'center', marginBottom: '0.75rem' }}>
          <div style={{ fontSize: '1rem', fontWeight: 700, color: '#334155' }}>{currentTrainingTimelineGraph.title}</div>
          <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
            {selectedTrainingTimelineTarget
              ? selectedTrainingTimelineTarget.is_average_profile
                ? `Average reward forensic profile built from ${selectedTrainingTimelineTarget.sample_count} episodes`
                : `Training episode ${selectedTrainingTimelineTarget.episode} timestep reward view`
              : 'Select a completed training episode to inspect timestep rewards'}
          </div>
        </div>
        {shouldShowTrainingTimelineKey && (
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '0.55rem 0.9rem',
              justifyContent: 'center',
              marginBottom: '0.85rem',
              padding: '0.65rem 0.75rem',
              backgroundColor: 'rgba(255,255,255,0.55)',
              border: '1px solid rgba(148, 163, 184, 0.18)',
              borderRadius: '12px',
            }}
          >
            {currentTrainingTimelineGraph.datasets.map((dataset) => (
              <div
                key={`training-key-${dataset.label}`}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.45rem',
                  color: '#334155',
                  fontSize: '0.82rem',
                  fontWeight: 600,
                }}
              >
                <span
                  style={{
                    width: '12px',
                    height: '12px',
                    borderRadius: '999px',
                    backgroundColor: dataset.borderColor,
                    border: '1px solid rgba(15, 23, 42, 0.18)',
                    flex: '0 0 auto',
                  }}
                />
                <span>{formatRewardTermLabel(dataset.label)}</span>
              </div>
            ))}
          </div>
        )}
        <div
          style={{
            marginBottom: '0.9rem',
            padding: '0.75rem 0.9rem',
            borderRadius: '12px',
            backgroundColor: selectedTrainingTimelineSummary.shadeColor,
            border: `1px solid ${selectedTrainingTimelineSummary.accentColor}`,
            textAlign: 'left',
          }}
        >
          <div style={{ fontWeight: 800, color: selectedTrainingTimelineSummary.accentColor }}>
            {selectedTrainingTimelineSummary.label}
          </div>
          <div style={{ color: '#475569', fontSize: '0.84rem', marginTop: '0.2rem' }}>
            {selectedTrainingTimelineSummary.detail}
          </div>
          {selectedTrainingTimelineSummary.highlightStart && selectedTrainingTimelineSummary.highlightEnd && (
            <div style={{ color: '#475569', fontSize: '0.8rem', marginTop: '0.2rem' }}>
              Highlighting timesteps {selectedTrainingTimelineSummary.highlightStart} to {selectedTrainingTimelineSummary.highlightEnd} to focus attention on the terminal region.
            </div>
          )}
        </div>
        <div
          style={{
            width: '100%',
            maxWidth: '100%',
            minWidth: 0,
            height: '320px',
            overflow: 'hidden',
          }}
        >
          <Line
            data={{
              labels: currentTrainingTimelineGraph.labels,
              datasets: currentTrainingTimelineGraph.datasets.map((dataset) => ({
                ...dataset,
                fill: false,
                tension: 0.18,
                pointRadius: 0,
                pointHoverRadius: 3,
                borderWidth: 2,
              })),
            }}
            plugins={trainingTimelinePlugins}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              animation: false,
              normalized: true,
              interaction: {
                intersect: false,
                mode: 'index',
              },
              plugins: {
                legend: {
                  display: currentTrainingTimelineGraph.datasets.length > 1,
                  position: 'bottom',
                },
              },
              scales: {
                x: {
                  title: { display: true, text: 'Timestep' },
                  ticks: {
                    autoSkip: true,
                    maxTicksLimit: 14,
                  },
                },
                y: { title: { display: true, text: 'Reward Contribution' } },
              },
            }}
          />
        </div>
      </div>
    )}
    <InsightsModal
      isOpen={!isSavedViewer && showTrainingInsights}
      onClose={() => setShowTrainingInsights(false)}
      title="Deterministic Training Insights"
      accentColor="#0f766e"
      description="Outcome-focused summaries derived from recent training episodes, using reward-term contrasts and terminal-window comparisons."
    >
      {renderInsightCards(trainingInsights, 'Training insights will appear after enough completed episodes are available.')}
      {renderBehaviorReportCards(
        trainingBehaviorReport,
        trainingBehaviorTags,
        'Training behavior metrics and tags will appear after enough completed episodes are available.'
      )}
    </InsightsModal>
    {!isSavedViewer && (trainingAblationReport || trainingAblationStatus === 'running' || trainingAblationStatus === 'error') && (
      <div style={{ ...sectionPanelStyle, marginTop: '1rem' }}>
        <h3 style={{ fontSize: '1.2rem', color: '#7c3aed', marginTop: 0 }}>Post-Training Reward Ablation</h3>
        <div style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: '0.8rem' }}>
          Re-scores one fixed batch of deterministic evaluation trajectories under native reward only, native plus each enabled term, and the full active reward configuration.
        </div>
        {trainingAblationStatus === 'running' && (
          <div style={{ color: '#475569', fontSize: '0.9rem' }}>
            Computing reward ablation study from the finished policy...
          </div>
        )}
        {trainingAblationStatus === 'error' && (
          <div style={{ color: '#b91c1c', fontSize: '0.9rem' }}>
            {trainingAblationReport?.error || 'Reward ablation study failed.'}
          </div>
        )}
        {Array.isArray(trainingAblationReport?.rows) && trainingAblationReport.rows.length > 0 && (
          <>
            <div style={{ color: '#64748b', fontSize: '0.82rem', marginBottom: '0.9rem' }}>
              Averaged over {trainingAblationReport.eval_episodes || 3} deterministic eval episodes using the same rollout samples for every reward config.
            </div>
            <div style={{ display: 'grid', gap: '0.7rem' }}>
              {trainingAblationReport.rows.map((row) => {
                const rewards = trainingAblationReport.rows.map((entry) => Number(entry.avg_eval_reward || 0));
                const maxReward = Math.max(...rewards, 1);
                const widthPct = Math.max(6, (Number(row.avg_eval_reward || 0) / maxReward) * 100);
                return (
                  <div
                    key={row.label}
                    style={{
                      border: '1px solid rgba(148, 163, 184, 0.18)',
                      borderRadius: '12px',
                      padding: '0.8rem',
                      backgroundColor: 'rgba(255,255,255,0.58)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                      <div style={{ fontWeight: 700, color: '#334155' }}>{row.label}</div>
                      <div style={{ fontFamily: 'ui-monospace, SFMono-Regular, monospace', color: '#0f172a' }}>
                        Avg Eval Reward: {Number(row.avg_eval_reward || 0).toFixed(2)}
                      </div>
                    </div>
                    <div
                      style={{
                        marginTop: '0.55rem',
                        height: '12px',
                        backgroundColor: '#e2e8f0',
                        borderRadius: '999px',
                        overflow: 'hidden',
                      }}
                    >
                      <div
                        style={{
                          width: `${widthPct}%`,
                          height: '100%',
                          background: 'linear-gradient(90deg, #6366f1, #22c55e)',
                        }}
                      />
                    </div>
                    <div style={{ marginTop: '0.4rem', fontSize: '0.78rem', color: '#64748b' }}>
                      Terms: {(row.term_keys || []).join(', ')}
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    )}
    {trainMode && (
      <ProgressBar isTraining={trainMode} runId={runId} />
  )}
    </div>


      {/* Playback Controls */}
      <div style={workspaceShellStyle}>
        <div style={workspaceHeaderStyle}>
          <div style={{ textAlign: 'left' }}>
            <h3 style={{ fontSize: '1.6rem', margin: 0, color: '#3b82f6' }}>🎮 Rollout Workspace</h3>
            <div style={workspaceMetaStyle}>
              This workspace handles playback, model swapping, saved rollout loading, frame inspection, and the rollout-only reward chart.
            </div>
          </div>
          <div style={statusBadgeStyle(isSavedViewer ? '#64748b' : isPaused ? '#10b981' : '#ef4444')}>
            <span>{isSavedViewer ? 'Saved Viewer' : isPaused ? 'Rollout Paused' : 'Rollout Live'}</span>
          </div>
        </div>
        
        <div style={{ ...sectionPanelStyle, display: 'grid', gap: '0.75rem', margin: '1rem 0' }}>
        {/* Row: “Load Model” label + file picker */}
        <div style={{display: 'flex', alignItems: 'center', gap: '0.75rem', justifyContent:'center'}}>
          <label htmlFor='loadModel' style={{ fontWeight: 1000 }}>Rollout Status:</label>
          <button
            onClick={() => togglePause()}
            disabled={isSavedViewer}
            style={{
              padding: '0.5rem 1.2rem',
              fontSize: '1rem',
              backgroundColor: isSavedViewer ? '#94a3b8' : isPaused ? '#10b981' : '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: isSavedViewer ? 'not-allowed' : 'pointer',
              boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
            }}
          >
            {isSavedViewer ? 'Saved Viewer' : isPaused ? '▶ Continue' : '⏸ Pause'}
          </button>
          <button
            onClick={setShowSavePopupToTrue}
            disabled={saving_rollouts}
            style={{
              backgroundColor: "#2563eb",
              color: "white",
              fontWeight: 600,
              padding: "8px 16px",
              border: "none",
              borderRadius: "6px",
              cursor: saving_rollouts ? "not-allowed" : "pointer",
              boxShadow: "0 1px 2px rgba(0,0,0,0.1)",
            }}
          >
            {saving_rollouts ? "Saving..." : "💾 Save Rollouts"}
          </button>
        </div>
        <div style={{display: 'flex', alignItems: 'center', gap: '0.75rem', justifyContent:'center'}}>
          <select
            value={selectedRolloutFile}
            onChange={(e) => setSelectedRolloutFile(e.target.value)}
            style={{ padding: '.35rem .5rem', minWidth: 180 }}
            title="Saved rollout JSON files from the rollouts folder"
          >
            <option value="">(Saved rollouts)</option>
            {rolloutFiles.map((fileName) => (
              <option key={fileName} value={fileName}>{fileName}</option>
            ))}
          </select>
          <button
            onClick={loadSavedRollouts}
            disabled={loadingSavedRollouts || !selectedRolloutFile}
            style={{
              backgroundColor: "#475569",
              color: "white",
              fontWeight: 600,
              padding: "8px 12px",
              border: "none",
              borderRadius: "6px",
              cursor: loadingSavedRollouts || !selectedRolloutFile ? "not-allowed" : "pointer",
              boxShadow: "0 1px 2px rgba(0,0,0,0.1)",
            }}
            title="Load a saved rollout JSON from the rollouts folder"
          >
            {loadingSavedRollouts ? "Loading..." : "Load Rollouts"}
          </button>
        </div>
        
        <div style={{ ...sectionPanelStyle, marginTop: '1rem', textAlign: 'center' }}>
          <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', color: '#334155' }}>Model Controls</h3>
          <div style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: '1rem' }}>
            Swap preview policies here without touching the training setup above.
          </div>
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
            disabled={isSavedViewer || !file || loading}
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
            value={modelSortOrder}
            onChange={(e) => setModelSortOrder(e.target.value)}
            disabled={isSavedViewer}
            style={{ padding: '.35rem .5rem', minWidth: 132, borderRadius: '8px', border: '1px solid #cbd5e1', backgroundColor: 'white' }}
            title="Sort saved models by creation time"
          >
              <option value="newest">Newest first</option>
              <option value="oldest">Oldest first</option>
            </select>
          <select
            id="serverModel"
            value={selectedServerModel}
            onChange={(e) => setSelectedServerModel(e.target.value)}
            disabled={isSavedViewer}
            style={{ padding: '.35rem .5rem', minWidth: 420, maxWidth: 520 }}
          >
              <option value="">(None — random policy)</option>
              {sortedServerModelRecords.map((model) => (
                <option
                  key={model.name}
                  value={model.name}
                  disabled={Boolean(model.env_name) && model.env_name !== envName}
                >
                  {`${model.env_name || 'Model'} | ${formatModelTimestamp(model.created_at)} | ${model.name}${
                    model.env_name && model.env_name !== envName ? ' | incompatible' : ''
                  }`}
                </option>
              ))}
            </select>
          <button
            onClick={deleteAllTempModels}
            disabled={isSavedViewer}
            onMouseEnter={() => setHoverOnDeleteAllTempButton(true)}
              onMouseLeave={() => setHoverOnDeleteAllTempButton(false)}
              style={{
                border: "none",
                background: "transparent",
                cursor: "pointer",
                fontSize: "1rem", // small text-sized
                padding: "0.2rem",
              }}
              title={"Delete ALl models in the temporary directory"}
            >
              {hoverOnDeleteAllTemp ? "🗑 Delete All" : "🗑"}
            </button>
          <button
            onClick={reloadTempModels}
            disabled={isSavedViewer}
            onMouseEnter={() => setHoveronReloadTempModels(true)}
              onMouseLeave={() => setHoveronReloadTempModels(false)}
              style={{
                border: "none",
                background: "transparent",
                cursor: "pointer",
                fontSize: "1rem", // small text-sized
                padding: "0.2rem",
              }}
              title="Reload all models in the temporary directory"
            >
              {hoveronReloadTempModels ? "↻ Reload" : "↻"}
            </button>
          <button
            onClick={loadServerModel}
            disabled={isSavedViewer || loading}
              style={{ padding: '.4rem .75rem',
                backgroundColor: !isUsingNone ? '#d1d5db' : '#10b981'
              }}

              title="Load the selected server model (or None)"
            >
              {loading ? "Loading..." : "Use Selection"}
            </button>
          <button
            onClick={useNone}
            disabled={isSavedViewer || loading}
              style={{ padding: '.4rem .75rem', 
                backgroundColor: isUsingNone ? '#d1d5db' : '#10b981'
              }}
              title="Clear model for this session (random rollout)"
            >
              Use None
            </button>
          </div>
          {!isSavedViewer && selectedServerModelRecord && (
            <div
              style={{
                marginTop: '0.85rem',
                padding: '0.8rem 0.9rem',
                borderRadius: '12px',
                border: '1px solid rgba(148, 163, 184, 0.18)',
                background: 'rgba(255,255,255,0.66)',
                color: '#475569',
                fontSize: '0.84rem',
                lineHeight: 1.5,
                textAlign: 'left',
              }}
            >
              <div style={{ fontWeight: 800, color: '#334155', marginBottom: '0.25rem' }}>
                {selectedServerModelRecord.env_name || 'Saved model'}
              </div>
              <div>Created: {formatModelTimestamp(selectedServerModelRecord.created_at)}</div>
              <div>Size: {formatModelSize(selectedServerModelRecord.size_bytes)}</div>
              <div>File: {selectedServerModelRecord.name}</div>
            </div>
          )}
        </div>
      </div>
      <div
        style={{
          ...sectionPanelStyle,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '0.75rem',
          flexWrap: 'wrap',
        }}
      >
        <button
          onClick={() => setRolloutWorkspaceViewIndex((prev) => (prev - 1 + rolloutWorkspaceViews.length) % rolloutWorkspaceViews.length)}
          style={{
            width: '36px',
            height: '36px',
            borderRadius: '999px',
            border: '1px solid #cbd5e1',
            backgroundColor: 'white',
            color: '#334155',
            fontWeight: 700,
          }}
          aria-label="Show previous rollout panel"
        >
          {'<'}
        </button>
        <div style={{ textAlign: 'center', flex: '1 1 240px' }}>
          <div style={{ fontSize: '1.05rem', fontWeight: 800, color: '#334155' }}>{currentRolloutWorkspaceView.title}</div>
          <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
            {rolloutWorkspaceViewIndex + 1} / {rolloutWorkspaceViews.length}
          </div>
        </div>
        <select
          value={String(rolloutWorkspaceViewIndex)}
          onChange={(event) => setRolloutWorkspaceViewIndex(Number(event.target.value))}
          style={{
            minWidth: '240px',
            padding: '0.45rem 0.6rem',
            borderRadius: '8px',
            border: '1px solid #cbd5e1',
            backgroundColor: 'white',
            color: '#334155',
            fontSize: '0.92rem',
          }}
          aria-label="Choose rollout workspace view"
        >
          {rolloutWorkspaceViews.map((view, index) => (
            <option key={view.key} value={index}>
              {view.title}
            </option>
          ))}
        </select>
        {!isSavedViewer && (
          <button
            onClick={() => setShowRolloutInsights((prev) => !prev)}
            style={{
              padding: '0.5rem 0.9rem',
              borderRadius: '999px',
              border: '1px solid rgba(37, 99, 235, 0.22)',
              backgroundColor: showRolloutInsights ? '#2563eb' : 'white',
              color: showRolloutInsights ? 'white' : '#2563eb',
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            {showRolloutInsights ? 'Hide Insights' : 'Show Insights'}
          </button>
        )}
        <button
          onClick={() => setRolloutWorkspaceViewIndex((prev) => (prev + 1) % rolloutWorkspaceViews.length)}
          style={{
            width: '36px',
            height: '36px',
            borderRadius: '999px',
            border: '1px solid #cbd5e1',
            backgroundColor: 'white',
            color: '#334155',
            fontWeight: 700,
          }}
          aria-label="Show next rollout panel"
        >
          {'>'}
        </button>
      </div>
      <InsightsModal
        isOpen={!isSavedViewer && showRolloutInsights}
        onClose={() => setShowRolloutInsights(false)}
        title="Deterministic Rollout Insights"
        accentColor="#2563eb"
        description="Recent rollout episodes are analyzed for terms that separate success from failure and for late-episode failure signatures."
      >
        {renderInsightCards(rolloutInsights, 'Rollout insights will appear after enough recent rollout episodes have been observed.')}
        {renderBehaviorReportCards(
          rolloutBehaviorReport,
          rolloutBehaviorTags,
          'Rollout behavior metrics and tags will appear after enough recent rollout episodes have been observed.'
        )}
      </InsightsModal>
      
      <div style={{ marginTop: "2rem", textAlign: "center" }}>
        <label htmlFor="rolloutSpeed" style={{ fontWeight: 600 }}>
          ⚡ Rollout Speed (FPS):
        </label>
        <br />
        <input
          id="rolloutSpeed"
          type="range"
          min="1"
          max="500"
          step="10"
          value={rolloutSpeed}
          onChange={(e) => updateRolloutSpeed(Number(e.target.value))}
          style={{ width: "200px", margin: "0.5rem" }}
        />
        <input
          type="number"
          min="1"
          step="10"
          value={rolloutSpeed}
          onChange={(e) => updateRolloutSpeed(Number(e.target.value))}
          style={{
            width: "70px",
            padding: "4px",
            border: "1px solid #d1d5db",
            borderRadius: "4px",
          }}
        />
      </div>
      {/* Playback Controls */}
      {currentRolloutWorkspaceView.key === 'visualization' && (
      <>
      <div style={sectionPanelStyle}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '0.75rem',
          flexWrap: 'wrap',
          marginBottom: '0.9rem',
        }}
      >
        <div style={{ textAlign: 'left', flex: '1 1 260px' }}>
          <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#334155' }}>Rollout Visualization</div>
          <div style={{ color: '#64748b', fontSize: '0.82rem', marginTop: '0.15rem' }}>
            Control capture cadence, playback speed, and rollout insight visibility from this header.
          </div>
          <div
            style={{
              marginTop: '0.5rem',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.45rem 0.7rem',
              borderRadius: '999px',
              background: activeModelName
                ? 'linear-gradient(180deg, rgba(14,165,233,0.16), rgba(37,99,235,0.12))'
                : 'linear-gradient(180deg, rgba(148,163,184,0.18), rgba(100,116,139,0.12))',
              border: activeModelName
                ? '1px solid rgba(37,99,235,0.25)'
                : '1px solid rgba(148,163,184,0.25)',
              color: '#334155',
              fontSize: '0.82rem',
              lineHeight: 1.35,
            }}
          >
            <span style={{ fontWeight: 800, color: activeModelName ? '#1d4ed8' : '#475569' }}>Active Model</span>
            <span style={{ fontFamily: 'ui-monospace, SFMono-Regular, monospace', color: '#0f172a' }}>
              {activeModelRecord?.name || activeModelName || 'None (random policy)'}
            </span>
            {activeModelRecord?.created_at && (
              <span style={{ color: '#64748b' }}>
                {formatModelTimestamp(activeModelRecord.created_at)}
              </span>
            )}
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <label htmlFor="visualizationStepInterval" style={{ fontWeight: 600, color: '#334155' }}>Every</label>
          <select
            id="visualizationStepInterval"
            value={stepInterval}
            onChange={(e) => changeNumberOfSteps(e.target.value)}
            disabled={isSavedViewer}
            style={{
              padding: "0.4rem 0.55rem",
              borderRadius: "8px",
              border: "1px solid #d1d5db",
              backgroundColor: 'white',
            }}
          >
            <option value={1}>1</option>
            <option value={2}>2</option>
            <option value={5}>5</option>
            <option value={8}>8</option>
            <option value={12}>12</option>
            <option value={18}>18</option>
          </select>
          <span style={{ color: '#64748b', fontSize: '0.85rem' }}>episodes</span>
          <label htmlFor="visualizationReplaySpeed" style={{ fontWeight: 600, color: '#334155', marginLeft: '0.25rem' }}>Playback</label>
          <input
            id="visualizationReplaySpeed"
            type="number"
            min="10"
            max="500"
            step="10"
            value={replayInterval}
            onChange={(e) => setReplayInterval(Number(e.target.value))}
            style={{
              width: '78px',
              padding: '0.4rem 0.45rem',
              border: '1px solid #d1d5db',
              borderRadius: '8px',
            }}
          />
          <span style={{ color: '#64748b', fontSize: '0.85rem' }}>ms/frame</span>
        </div>
      </div>
      <p style={{ fontSize: '1rem' }}>
        Simulating Episode <strong style={{ color: '#0ea5e9' }}>{selectedVisualizationEpisode ?? episodeNumForSimulation}</strong>
      </p>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '0.75rem' }}>
        <label htmlFor="visualizationEpisode" style={{ fontWeight: 600 }}>Captured rollout episode</label>
        <select
          id="visualizationEpisode"
          value={selectedVisualizationEpisode ?? ''}
          onChange={(event) => setSelectedVisualizationEpisode(Number(event.target.value))}
          disabled={availableVisualizationEpisodes.length === 0}
          style={{ padding: '.4rem .55rem', minWidth: 220 }}
        >
          {availableVisualizationEpisodes.length === 0 ? (
            <option value="">No captured rollout videos yet</option>
          ) : (
            availableVisualizationEpisodes.map((episode) => (
              <option key={`visualization-episode-${episode}`} value={episode}>
                Episode {episode}
              </option>
            ))
          )}
        </select>
      </div>
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
      </div>

      {false && (
      <>
      {/* Playback Speed Slider */}
      <div style={{ ...sectionPanelStyle, marginTop: '1rem', textAlign: 'center' }}>
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
      </>
      )}
      {/*<RolloutSlideshow/>*/}
      {visualizationFrames && visualizationFrames.length > 0 && (
      <div style={{ ...sectionPanelStyle, marginTop: '1rem', textAlign: 'center' }}>
        <img
          src={`data:image/jpeg;base64,${visualizationFrames[currentFrame]}`}
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
      </>
      )}

    {currentRolloutWorkspaceView.key === 'reward_chart' && (
    <div style={{ ...sectionPanelStyle, marginTop: '1rem' }}>
      <h3 style={{ fontSize: '1.6rem', color: '#6366f1', marginTop: 0 }}>📈 Rollout Reward Chart</h3>
      <div style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: '0.8rem' }}>
        This chart is reserved for rollout episodes only. Training rewards stay in the workspace above.
      </div>
      <p>
        Episode <strong>{episodeInfo.episode}</strong>, Reward:{' '}
        <strong style={{ color: '#10b981' }}>{episodeInfo.reward}</strong>
      </p>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '0.75rem',
          marginBottom: '0.85rem',
          flexWrap: 'wrap',
        }}
      >
        <button
          onClick={() => setRolloutGraphIndex((prev) => (prev - 1 + rolloutGraphDefinitions.length) % rolloutGraphDefinitions.length)}
          disabled={rolloutGraphDefinitions.length <= 1}
          style={{
            width: '32px',
            height: '32px',
            borderRadius: '999px',
            border: '1px solid #cbd5e1',
            backgroundColor: 'white',
            cursor: rolloutGraphDefinitions.length <= 1 ? 'not-allowed' : 'pointer',
            fontWeight: 700,
            color: '#334155',
          }}
          aria-label="Show previous rollout graph"
        >
          {'<'}
        </button>
        <div style={{ textAlign: 'center', flex: '1 1 240px' }}>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#334155' }}>{currentRolloutGraph.title}</div>
          <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
            {rolloutGraphDefinitions.length > 0 ? `${rolloutGraphIndex + 1} / ${rolloutGraphDefinitions.length}` : 'No rollout data yet'}
          </div>
        </div>
        <select
          value={String(rolloutGraphIndex)}
          onChange={(e) => setRolloutGraphIndex(Number(e.target.value))}
          style={{
            minWidth: '240px',
            padding: '0.45rem 0.6rem',
            borderRadius: '8px',
            border: '1px solid #cbd5e1',
            backgroundColor: 'white',
            color: '#334155',
            fontSize: '0.92rem',
          }}
          aria-label="Choose rollout reward term graph"
        >
          {rolloutGraphDefinitions.map((graph, index) => (
            <option key={graph.title} value={index}>
              {graph.title}
            </option>
          ))}
        </select>
        <button
          onClick={() => setRolloutGraphIndex((prev) => (prev + 1) % rolloutGraphDefinitions.length)}
          disabled={rolloutGraphDefinitions.length <= 1}
          style={{
            width: '32px',
            height: '32px',
            borderRadius: '999px',
            border: '1px solid #cbd5e1',
            backgroundColor: 'white',
            cursor: rolloutGraphDefinitions.length <= 1 ? 'not-allowed' : 'pointer',
            fontWeight: 700,
            color: '#334155',
          }}
          aria-label="Show next rollout graph"
        >
          {'>'}
        </button>
      </div>
      <div style={{ width: '100%', maxWidth: '1000px', margin: '0 auto', overflowX: 'auto', paddingBottom: '0.5rem' }}>
        <div
          style={{
            width: `${Math.max(
              rolloutChartMinWidth,
              Math.ceil(Math.max(1, rollouts.length) / rolloutChartBucketSize) * rolloutChartBucketSize * rolloutChartPointWidth
            )}px`,
            height: '300px',
          }}
        >
          <Line
            data={{
              labels: currentRolloutGraph.labels,
              datasets: currentRolloutGraph.datasets.map((dataset) => ({
                ...dataset,
                fill: false,
                tension: 0.2,
                pointRadius: 0,
                pointHoverRadius: 3,
                borderWidth: 2,
              })),
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              animation: false,
              normalized: true,
              interaction: {
                intersect: false,
                mode: 'index',
              },
              scales: {
                x: { title: { display: true, text: "Episode" } },
                y: { title: { display: true, text: "Reward" } },
              },
            }}
          />
        </div>
      </div>
    </div>
    )}
    {currentRolloutWorkspaceView.key === 'temporal_breakdown' && (
    <div style={{ ...sectionPanelStyle, marginTop: '1rem' }}>
      <h3 style={{ fontSize: '1.35rem', color: '#0f766e', marginTop: 0 }}>Temporal Reward Breakdown</h3>
      <div style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: '0.8rem' }}>
        Inspect reward contributions inside one episode to see which term spikes right before failure and which term dominates each timestep.
      </div>
        <div
          style={{
            display: 'flex',
            gap: '0.75rem',
            alignItems: 'center',
          justifyContent: 'center',
          flexWrap: 'wrap',
            marginBottom: '0.9rem',
          }}
        >
        <label htmlFor="rolloutTimelineMode" style={{ fontWeight: 600 }}>Mode</label>
        <select
          id="rolloutTimelineMode"
          value={rolloutTimelineMode}
          onChange={(event) => setRolloutTimelineMode(event.target.value)}
          style={{ padding: '0.4rem 0.55rem', minWidth: 180 }}
        >
          <option value="episode">Single Episode</option>
          <option value="average">Average By Type</option>
        </select>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
          <span style={{ fontWeight: 600 }}>Episode Type</span>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
            <input
              type="radio"
              name="rolloutTimelineOutcome"
              value="all"
              checked={rolloutTimelineOutcomeFilter === 'all'}
              onChange={(event) => setRolloutTimelineOutcomeFilter(event.target.value)}
            />
            All
          </label>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
            <input
              type="radio"
              name="rolloutTimelineOutcome"
              value="success"
              checked={rolloutTimelineOutcomeFilter === 'success'}
              onChange={(event) => setRolloutTimelineOutcomeFilter(event.target.value)}
            />
            Successful ({rolloutOutcomeCounts.success})
          </label>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
            <input
              type="radio"
              name="rolloutTimelineOutcome"
              value="failure"
              checked={rolloutTimelineOutcomeFilter === 'failure'}
              onChange={(event) => setRolloutTimelineOutcomeFilter(event.target.value)}
            />
            Failed ({rolloutOutcomeCounts.failure})
          </label>
        </div>
        <label htmlFor="timelineEpisode" style={{ fontWeight: 600 }}>Episode</label>
        <select
          id="timelineEpisode"
          value={selectedTimelineEpisode ?? ''}
          onChange={(event) => setSelectedTimelineEpisode(Number(event.target.value))}
          style={{ padding: '0.4rem 0.55rem', minWidth: 160 }}
          disabled={rolloutTimelineMode === 'average' || filteredRollouts.length === 0}
        >
          {orderedFilteredRollouts.length === 0 ? (
            <option value="">No episodes yet</option>
          ) : (
            orderedFilteredRollouts.map((entry) => (
              <option key={`episode-${entry.episode}`} value={entry.episode}>
                Episode {entry.episode}
              </option>
            ))
          )}
        </select>
        <label htmlFor="timelineView" style={{ fontWeight: 600 }}>View</label>
        <select
          id="timelineView"
          value={String(timelineGraphIndex)}
          onChange={(event) => setTimelineGraphIndex(Number(event.target.value))}
          style={{ padding: '0.4rem 0.55rem', minWidth: 280 }}
          disabled={timelineGraphDefinitions.length === 0}
        >
          {timelineGraphDefinitions.length === 0 ? (
            <option value="0">No reward history yet</option>
          ) : (
            timelineGraphDefinitions.map((graph, index) => (
              <option key={graph.title} value={index}>
                {graph.title}
              </option>
            ))
          )}
        </select>
      </div>
        <div style={{ textAlign: 'center', marginBottom: '0.75rem' }}>
          <div style={{ fontSize: '1rem', fontWeight: 700, color: '#334155' }}>{currentTimelineGraph.title}</div>
          <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
            {selectedTimelineTarget
              ? selectedTimelineTarget.is_average_profile
              ? `Average reward forensic profile built from ${selectedTimelineTarget.sample_count} episodes`
              : `Episode ${selectedTimelineTarget.episode} timestep reward view`
            : 'Select an episode to inspect timestep rewards'}
        </div>
      </div>
      {shouldShowRolloutTimelineKey && (
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '0.55rem 0.9rem',
            justifyContent: 'center',
            marginBottom: '0.85rem',
            padding: '0.65rem 0.75rem',
            backgroundColor: 'rgba(255,255,255,0.55)',
            border: '1px solid rgba(148, 163, 184, 0.18)',
            borderRadius: '12px',
          }}
        >
          {currentTimelineGraph.datasets.map((dataset) => (
            <div
              key={`rollout-key-${dataset.label}`}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.45rem',
                color: '#334155',
                fontSize: '0.82rem',
                fontWeight: 600,
              }}
            >
              <span
                style={{
                  width: '12px',
                  height: '12px',
                  borderRadius: '999px',
                  backgroundColor: dataset.borderColor,
                  border: '1px solid rgba(15, 23, 42, 0.18)',
                  flex: '0 0 auto',
                }}
              />
              <span>{formatRewardTermLabel(dataset.label)}</span>
            </div>
          ))}
        </div>
      )}
      <div
        style={{
          marginBottom: '0.9rem',
          padding: '0.75rem 0.9rem',
          borderRadius: '12px',
          backgroundColor: selectedRolloutTimelineSummary.shadeColor,
          border: `1px solid ${selectedRolloutTimelineSummary.accentColor}`,
          textAlign: 'left',
        }}
      >
        <div style={{ fontWeight: 800, color: selectedRolloutTimelineSummary.accentColor }}>
          {selectedRolloutTimelineSummary.label}
        </div>
        <div style={{ color: '#475569', fontSize: '0.84rem', marginTop: '0.2rem' }}>
          {selectedRolloutTimelineSummary.detail}
        </div>
        {selectedRolloutTimelineSummary.highlightStart && selectedRolloutTimelineSummary.highlightEnd && (
          <div style={{ color: '#475569', fontSize: '0.8rem', marginTop: '0.2rem' }}>
            Highlighting timesteps {selectedRolloutTimelineSummary.highlightStart} to {selectedRolloutTimelineSummary.highlightEnd} to focus attention on the terminal region.
          </div>
        )}
      </div>
      <div
        style={{
          width: '100%',
          maxWidth: '100%',
          minWidth: 0,
          height: '320px',
          overflow: 'hidden',
        }}
      >
        <Line
          data={{
            labels: currentTimelineGraph.labels,
            datasets: currentTimelineGraph.datasets.map((dataset) => ({
              ...dataset,
              fill: false,
              tension: 0.18,
              pointRadius: 0,
              pointHoverRadius: 3,
              borderWidth: 2,
            })),
          }}
          plugins={rolloutTimelinePlugins}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            normalized: true,
            onClick: (_, elements) => {
              if (!elements || elements.length === 0) return;
              if (rolloutTimelineMode !== 'episode') return;
              const nextIndex = elements[0].index;
              setSelectedTimelineStep(nextIndex + 1);
            },
            interaction: {
              intersect: false,
              mode: 'index',
            },
            plugins: {
              legend: {
                display: currentTimelineGraph.datasets.length > 1,
                position: 'bottom',
              },
            },
            scales: {
              x: {
                title: { display: true, text: 'Timestep' },
                ticks: {
                  autoSkip: true,
                  maxTicksLimit: 14,
                },
              },
              y: { title: { display: true, text: 'Reward Contribution' } },
            },
            }}
          />
        </div>
      <div style={{ marginTop: '1rem', display: 'grid', gridTemplateColumns: 'minmax(260px, 1fr) minmax(260px, 1fr)', gap: '1rem' }}>
        <div
          style={{
            borderRadius: '12px',
            border: '1px solid rgba(148, 163, 184, 0.18)',
            background: 'rgba(255,255,255,0.55)',
            padding: '0.9rem',
          }}
        >
          <div style={{ fontWeight: 800, color: '#334155', marginBottom: '0.45rem' }}>Linked Rollout Frame</div>
          <div style={{ color: '#64748b', fontSize: '0.82rem', marginBottom: '0.7rem' }}>
            Click a timestep on the chart to jump to the matching captured frame when that episode has video.
          </div>
          {rolloutTimelineMode === 'average' ? (
            <div style={{ color: '#64748b', fontSize: '0.85rem' }}>
              Frame linkage is only available in `Single Episode` mode.
            </div>
          ) : selectedEpisodeFrames.length === 0 ? (
            <div style={{ color: '#64748b', fontSize: '0.85rem' }}>
              No captured video exists for this episode. Video is only recorded every configured capture interval.
            </div>
          ) : (
            <>
              <div style={{ color: '#334155', fontWeight: 700, marginBottom: '0.55rem' }}>
                Timestep {selectedTimelineStep ?? 1} / {selectedEpisodeFrames.length}
              </div>
              <img
                src={`data:image/jpeg;base64,${selectedEpisodeFrames[linkedSelectedFrameIndex]}`}
                alt={`episode ${selectedTimelineEpisode} timestep ${selectedTimelineStep}`}
                style={{
                  width: '100%',
                  maxWidth: '460px',
                  borderRadius: '12px',
                  border: '2px solid rgba(14, 165, 233, 0.35)',
                  boxShadow: '0 6px 18px rgba(15, 23, 42, 0.12)',
                }}
              />
              <input
                type="range"
                min="1"
                max={selectedEpisodeFrames.length}
                step="1"
                value={Math.max(1, selectedTimelineStep || 1)}
                onChange={(event) => setSelectedTimelineStep(Number(event.target.value))}
                style={{ width: '100%', marginTop: '0.8rem' }}
              />
            </>
          )}
        </div>
        <div
          style={{
            borderRadius: '12px',
            border: '1px solid rgba(148, 163, 184, 0.18)',
            background: 'rgba(255,255,255,0.55)',
            padding: '0.9rem',
            textAlign: 'left',
          }}
        >
          <div style={{ fontWeight: 800, color: '#334155', marginBottom: '0.45rem' }}>Exact Reward Structure At Selected Timestep</div>
          <div style={{ color: '#64748b', fontSize: '0.82rem', marginBottom: '0.7rem' }}>
            This is the reward forensic snapshot for the currently selected timestep.
          </div>
          {!selectedTimelineStepEntry ? (
            <div style={{ color: '#64748b', fontSize: '0.85rem' }}>
              Select a single rollout episode and click a timestep on the chart.
            </div>
          ) : (
            <>
              <div style={{ color: '#334155', fontWeight: 700, marginBottom: '0.5rem' }}>
                Timestep {selectedTimelineStepEntry.step}
              </div>
              <div style={{ color: '#0f766e', fontWeight: 700, marginBottom: '0.7rem' }}>
                Total reward: {Number(selectedTimelineStepEntry.reward_breakdown?.total ?? selectedTimelineStepEntry.reward ?? 0).toFixed(3)}
              </div>
              {Object.entries(selectedTimelineStepEntry.reward_breakdown || {}).filter(([key]) => key !== 'total').map(([key, value]) => (
                <div
                  key={`selected-breakdown-${key}`}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    gap: '0.75rem',
                    padding: '0.22rem 0',
                    borderBottom: '1px solid rgba(148, 163, 184, 0.12)',
                  }}
                >
                  <span style={{ color: '#475569' }}>{key}</span>
                  <span style={{ fontFamily: 'ui-monospace, SFMono-Regular, monospace' }}>{Number(value).toFixed(3)}</span>
                </div>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
    )}
    </div>
  </div>
);


}

export default RolloutWindow
