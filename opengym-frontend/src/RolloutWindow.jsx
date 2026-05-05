import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import {Line} from 'react-chartjs-2'
import SetPathPopup from './SetPathPopup'
import SaveRolloutPopup from './RolloutPopup'
import ProgressBar from './ProgressBar'
import axios  from 'axios'
import {Chart as ChartJS, LineElement, CategoryScale, LinearScale, PointElement} from 'chart.js'

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement);

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
  const [frames, setFrames] = useState([]);
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
  const [trainingRewardBreakdown, setTrainingRewardBreakdown] = useState({});
  const [trainingRewardBreakdownMean, setTrainingRewardBreakdownMean] = useState({});
  const [trainingGraphIndex, setTrainingGraphIndex] = useState(0);
  const [rolloutGraphIndex, setRolloutGraphIndex] = useState(0);
  const [rolloutRewardBreakdown, setRolloutRewardBreakdown] = useState({});
  const [latestRolloutRawTerms, setLatestRolloutRawTerms] = useState({});
  const [rewardLogs, setRewardLogs] = useState([]);

  const [envName, setEnvName] = useState(initialEnvName);
  const [isPaused, setIsPaused] = useState(false);

  // the FPS of rollout, default is 20FPS (delay = 1/20 = 0.05 seconds)
  const [rolloutSpeed, setRolloutSpeed] = useState(20);
  const [showSavePopup, setShowSavePopup] = useState(false);
  const setShowSavePopupToTrue = () => setShowSavePopup(true);
  const closeShowSavePopup = () => setShowSavePopup(false);

  const isPausedRef = useRef(false);
  const [sessionId, setSessionId] = useState(null);
  // whether or not the current rollout (all the rewards) is being saved
  const [saving_rollouts, setSavingRollouts] = useState(false);
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
  const rewardLogLimit = 30;
  const rolloutChartMinWidth = 600;
  const rolloutChartPointWidth = 36;
  const rolloutChartBucketSize = 25;

  const appendRewardLog = useCallback((entry) => {
    setRewardLogs((prev) => [entry, ...prev.slice(0, rewardLogLimit - 1)]);
  }, []);

  // Upload the files logistics:
  const [serverModels, setServerModels] = useState([]);
  const [selectedServerModel, setSelectedServerModel] = useState(""); // "" = None
  const [rolloutFiles, setRolloutFiles] = useState([]);
  const [selectedRolloutFile, setSelectedRolloutFile] = useState("");
  const [showPopup, setShowPopup] = useState(false);
  const [file, setFile] = useState(null);
  // whether is using default policy or not
  const [isUsingNone, setIsUsingNone] = useState(true);
  const [loading, setLoading] = useState(false);
  const [loadingSavedRollouts, setLoadingSavedRollouts] = useState(false);

  //set whether model parent directory file path is copied
  const [filePathCopied, setFilePathCopied] = useState(false);
  const [hoverOnFilePathButton, setHoverOnFilePathButton] = useState(false);
  const [hoverOnDeleteAllTemp, setHoverOnDeleteAllTempButton] = useState(false);

  const handleEnvChange = (e) => {
    if (isSavedViewer) return;
    setEnvName(e.target.value)
  }

  const hydrateLoadedRollouts = useCallback((loadedRollouts) => {
    const safeRollouts = Array.isArray(loadedRollouts) ? loadedRollouts : [];
    setRollouts(safeRollouts);
    setTrainingRollouts([]);
    setFrames([]);
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
      };

      return [...prev, { ...seededTerm, [field]: nextValue }];
    });
    setRewardConfigDirty(true);
    setRewardConfigStatus("Unsaved reward changes.");
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
      const nextBreakdown = computeBreakdownFromRawTerms(terms, entry.reward_raw_terms || {});
      if (!nextBreakdown) {
        return entry;
      }
      return {
        ...entry,
        reward: nextBreakdown.total,
        reward_breakdown: nextBreakdown,
      };
    });
  }, [computeBreakdownFromRawTerms]);

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
      const response = await axios.post("/reward_config", {
        run_id: runId,
        env_name: envName,
        terms: rewardConfig.map((term) => ({
          key: term.key,
          weight: Number(term.weight),
          enabled: Boolean(term.enabled),
        })),
      });
      const nextTerms = response.data.terms || [];
      setRewardConfig(nextTerms);
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
  }, [applyRewardConfigToRollouts, computeBreakdownFromRawTerms, envName, isSavedViewer, latestRolloutRawTerms, rewardConfig, rollouts, runId]);
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
        setFrozenPath(`models/ppo_model_${envName}_${timestamp}.zip`);
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
      await axios.post("/rollout_speed", { "run_id": runId, "fps":newSpeed,"delay": 1.0 / newSpeed });
    } catch (e) {
      console.error("Failed to set rollout speed:", e);
    }
  };
  const saveTrainingPath = async (path, device, nextTrainingHyperparams) => {
    if (!runId) {
      console.warn("Run ID not set yet, cannot pause/resume");
      return;
    }
    try {
      // Set Train path FIRST
      // THEN SET THE TRAIN MODE AND toggle pause
      // persist to backend (example endpoint)
      await axios.post("/set_training_dir", {
        "run_id": runId,
        "train_dir_path": path,
        "device": device,
        "training_hyperparams": nextTrainingHyperparams,
      });
      setTrainingPath(path);
      setTrainingHyperparams(nextTrainingHyperparams);
      closePathPopup();

      toggleTrainPauseTogether();

      
    } catch (e) {
      console.error("Failed to set training path:", e);
      // optionally show a toast here
    }
  };
  const deleteAllTempModels = () => {
    // delete all the models in the temporary directory
    axios.post("/delete_all_temp_models", { "run_id": runId });
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
      await axios.post("/change_number_of_steps", { "run_id": runId, "number_of_steps": steps });
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
    await axios.post("/pause_rollout", { session_id: sessionId, paused: newState });
    isPausedRef.current = newState;
    setIsPaused(newState);
  };

  useEffect(() => {
    if (isSavedViewer) {
      setRewardConfig([]);
      setSupportsCustomReward(false);
      setRewardConfigDirty(false);
      setRewardConfigLoading(false);
      setRewardConfigStatus("Saved rollout viewer. Reward terms shown below come from the loaded JSON.");
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
    if (rolloutGraphDefinitions.length === 0) {
      setRolloutGraphIndex(0);
      return;
    }
    setRolloutGraphIndex((prev) => prev % rolloutGraphDefinitions.length);
  }, [rolloutGraphDefinitions.length]);

  useEffect(() => {
    let retryTimeout;
    // console.log("Fetching models from server..."); // DEBUG:FRONTEND
    const fetchModels = async () => {
      try {
        const res = await axios.get("/models");
        // Get the models that currently exist
        // console.log("Available models: ", res); // DEBUG:FRONTEND
        setServerModels(res.data.models || []);
      } catch (e) {
        console.error("List the models process has failed: Will retry in 5 seconds");
        //retry timeout = 5 seconds
        retryTimeout = setTimeout(fetchModels, 5000);
      }
    };
    const fetchRolloutFiles = async () => {
      try {
        const res = await axios.get("/rollouts_files");
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
        const returnData = await axios.get("/unique_run_id");
        if (!cancelled) {
          setRunId(returnData.data.run_id);
          attempt = 0; // reset the backoff attempt after success
        }
      } catch (error) {
        if (!cancelled) {
          // if intential cancel, don't retry
          if (axios.isCancel?.(error) || error?.name === "CanceledError") return;
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
        const response = await axios.get("/reward_config", {
          params: { run_id: runId, env_name: envName },
        });
        if (cancelled) return;
        setRewardConfig(response.data.terms || []);
        setSupportsCustomReward(Boolean(response.data.supports_custom_reward));
        setRewardConfigDirty(false);
        setRewardConfigStatus(
          response.data.supports_custom_reward
            ? "Editing applies to rollout and training live."
            : "Only native Gym reward is available for this environment right now."
        );
      } catch (error) {
        if (cancelled) return;
        console.error("Failed to fetch reward config:", error);
        setRewardConfigStatus("Failed to load reward settings.");
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
    if (!isActive) return;

    onSidebarStateChange({
      envName,
      rewardConfig,
      rewardConfigDirty,
      rewardConfigLoading,
      rewardConfigStatus,
      supportsCustomReward,
      latestTrainingBreakdown: trainingRewardBreakdown,
      latestTrainingMeanBreakdown: trainingRewardBreakdownMean,
      latestRolloutBreakdown: rolloutRewardBreakdown,
      rewardLogs,
      onTermChange: updateRewardTerm,
      onSaveConfig: saveRewardConfig,
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
    trainingRewardBreakdown,
    trainingRewardBreakdownMean,
    rolloutRewardBreakdown,
    rewardLogs,
    updateRewardTerm,
    saveRewardConfig,
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
        const url = `ws://localhost:8000/ws/rollout?runid=${runId}&env=${envName}&train=${trainMode}&train_steps=${trainSteps}`
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
            appendRewardLog({
              source: 'training',
              label: `Step ${data.step}`,
              total: data.reward_breakdown?.total ?? data.reward ?? 0,
              breakdown: data.reward_breakdown || {},
              at: data.ts ? new Date(data.ts * 1000).toLocaleTimeString() : 'training update',
            });
          } else {
            console.log("Received Episode data: ", data.type, data);
            // if data.type is not session
            if(data.ep_frames.length > 0){
              setFrames(data.ep_frames);        // store all frames
              setCurrentFrame(0);            // start at first frame
            }
            // console.log("Episode: ", data.episode, "   Reward: ", data.reward); // DEBUG:FRONTEND
            // console.log("Frames received length: ", data.ep_frames.length); // DEBUG:FRONTEND
            // console.log("Data sim frame episode number: ", data.sim_frame_episode_number); // DEBUG:FRONTEND
            if(data.sim_frame_episode_number) {
              setEpisodeNumForSimulation(data.sim_frame_episode_number);
            }
            //setIsPlaying(true); <- playback controlled by isPlaying var           // start playback automatically
            // don't need all the other information
            const newData = {
              reward: data.reward,
              episode: data.episode,
              reward_breakdown: data.reward_breakdown || {},
              reward_raw_terms: data.reward_raw_terms || {},
            };
            if (!trainMode) {
              setEpisodeInfo({ episode: data.episode, reward: data.reward });
              setRolloutRewardBreakdown(data.reward_breakdown || {});
              setLatestRolloutRawTerms(data.reward_raw_terms || {});
              setRollouts((prev) => [newData, ...prev]);
              appendRewardLog({
                source: 'rollout',
                label: `Episode ${data.episode}`,
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
      setTrainingRewardBreakdown({});
      setTrainingRewardBreakdownMean({});
      setRolloutRewardBreakdown({});
      setRewardLogs([]);
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
  }, [envName, isSavedViewer, trainMode, runId]);

  // This is the useEffect for the frame Data from the video
  useEffect(() => {
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
    if (trainMode) return;
    openPathPopup();
    // process the trainMode variable WITHIN the popup (i.e. after popup closes)
  }

  const stopTraining = async () => {
    if (!runId || stoppingTraining) return;
    setStoppingTraining(true);
    try {
      await axios.post("/stop_training", { run_id: runId });
      setTrainMode(false);
    } catch (e) {
      console.error("Failed to stop training:", e);
    } finally {
      setStoppingTraining(false);
    }
  };
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
        run_id: runId,
        model_name: selectedServerModel,
      });
    } finally {
      setLoading(false);
    }
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
      const res = await axios.post("/save_rollouts_data", { run_id: runId, rollout_filename: filename, rollouts: rollouts});
      const filesRes = await axios.get("/rollouts_files");
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
    }
  };

  const loadSavedRollouts = async () => {
    if (!selectedRolloutFile) {
      alert("No rollout file selected.");
      return;
    }
    setLoadingSavedRollouts(true);
    try {
      const res = await axios.post("/load_rollouts_data", {
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
      const up = await axios.post("/upload_model", form);
      const modelName = up.data?.model_name; // backend should return stored filename
      if (modelName) {
        await axios.post("/load_model", { run_id: runId, model_name: modelName });
      }
    } finally {
      setLoading(false);
      setFile(null);
    }
  };

  const getRootSavedModelsLink = async() => {
    const result = await axios.get("/get_model_path");
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
    background: 'linear-gradient(145deg, #f0f9ff, #e0e7ff)',
    borderRadius: '18px',
    border: '1px solid rgba(148, 163, 184, 0.18)',
    boxShadow: '0 8px 20px rgba(0,0,0,0.1)',
    padding: '1.35rem',
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
    color: '#64748b',
    fontSize: '0.9rem',
    maxWidth: '52rem',
    lineHeight: 1.5,
  };

  const sectionPanelStyle = {
    background: 'rgba(255,255,255,0.55)',
    borderRadius: '14px',
    border: '1px solid rgba(148, 163, 184, 0.18)',
    padding: '1rem',
    marginTop: '1rem',
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
      
    <div style={{ ...sectionPanelStyle, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
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
        defaultPath={frozenPath !== null ? frozenPath : `models/ppo_model_${envName}_${timestamp}.zip`}
        defaultHyperparams={trainingHyperparams}
        onConfirm={saveTrainingPath}
        onClose={closePathPopup}
      />
      <SaveRolloutPopup
        runId={runId}
        isOpen={showSavePopup}
        onConfirm={handleSave}
        onClose={closeShowSavePopup}
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
            id="serverModel"
            value={selectedServerModel}
            onChange={(e) => setSelectedServerModel(e.target.value)}
            disabled={isSavedViewer}
            style={{ padding: '.35rem .5rem', minWidth: 260 }}
          >
              <option value="">(None — random policy)</option>
              {serverModels.map(m => (
                <option key={m} value={m}>{m}</option>
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
        </div>
      </div>
      
      {/* Playback Controls */}
      <div style={sectionPanelStyle}>
      <p style={{ fontSize: '1rem', marginTop: 0 }}>
        Simulating per every {" "}
        <select 
          value={stepInterval}
          onChange={(e) => changeNumberOfSteps(e.target.value)}
          disabled={isSavedViewer}
          style={{
            padding: "4px",
            borderRadius: "4px",
            border: "1px solid #d1d5db",
            marginLeft: "0.25rem",
          }}>
            <option value={1}>1</option>
            <option value={2}>2</option>
            <option value={5}>5</option>
            <option value={8}>8</option>
            <option value={12}>12</option>
            <option value={18}>18</option>
        </select> {" "}
        steps
      </p>
      <p style={{ fontSize: '1rem' }}>
        Simulating Episode <strong style={{ color: '#0ea5e9' }}>{episodeNumForSimulation}</strong>
      </p>
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
    {/*<RolloutSlideshow/>*/}
    {frames && frames.length > 0 && (
      <div style={{ ...sectionPanelStyle, marginTop: '1rem', textAlign: 'center' }}>
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

    <div style={{ ...sectionPanelStyle, marginTop: '1rem' }}>
      <h3 style={{ fontSize: '1.6rem', color: '#6366f1', marginTop: 0 }}>📈 Rollout Reward Chart</h3>
      <div style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: '0.8rem' }}>
        This chart is reserved for rollout episodes only. Training rewards stay in the workspace above.
      </div>
      <p>
        Episode <strong>{episodeInfo.episode}</strong>, Reward:{' '}
        <strong style={{ color: '#10b981' }}>{episodeInfo.reward}</strong>
      </p>
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
    </div>
  </div>
);


}

export default RolloutWindow
