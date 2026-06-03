import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

const SetPathPopup = ({
  isOpen,
  defaultPath = "./models/ppo_model",
  defaultTrainSteps = 1000,
  defaultHyperparams = {},
  onConfirm,
  onClose,
}) => {
  const [path, setPath] = useState(defaultPath);
  const [device, setDevice] = useState("cuda");
  const [trainSteps, setTrainSteps] = useState(defaultTrainSteps);
  const [hyperparams, setHyperparams] = useState({
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
  const inputRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      setPath(defaultPath);
      setDevice("cuda");
      setTrainSteps(defaultTrainSteps);
      setHyperparams((prev) => ({ ...prev, ...defaultHyperparams }));
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [isOpen, defaultPath, defaultHyperparams, defaultTrainSteps]);

  useEffect(() => {
    if (!isOpen) {
      return undefined;
    }
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [isOpen]);

  const updateHyperparam = (key, value, cast = Number) => {
    setHyperparams((prev) => ({
      ...prev,
      [key]: cast === Number ? Number(value) : value,
    }));
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter") onConfirm?.(path, device, trainSteps, hyperparams);
    if (e.key === "Escape") onClose?.();
  };

  if (!isOpen) return null;

  return createPortal(
    <div className="popup-backdrop" onClick={onClose}>
      <div className="popup-card popup-card--form" onClick={(e) => e.stopPropagation()}>
        <div className="popup-header">
          <div>Set Training Path</div>
          <button className="popup-close" onClick={onClose} aria-label="Close">x</button>
        </div>

        <div className="popup-body popup-body--scroll">
          <div className="popup-hero-field">
            <label className="popup-label popup-label--hero">Training Steps</label>
            <div className="popup-hero-copy">
              Set the training budget first. This controls how long the backend PPO run will train before it stops.
            </div>
            <div className="popup-hero-controls">
              <input
                className="popup-slider popup-slider--hero"
                type="range"
                min="1000"
                max="100000"
                step="1000"
                value={trainSteps}
                onChange={(e) => setTrainSteps(Number(e.target.value))}
              />
              <input
                className="popup-input popup-input--hero-number"
                type="number"
                min="1000"
                max="100000"
                step="1000"
                value={trainSteps}
                onChange={(e) => setTrainSteps(Number(e.target.value))}
              />
            </div>
          </div>

          <label className="popup-label">Training output filename</label>
          <input
            ref={inputRef}
            className="popup-input"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={defaultPath}
          />
          <div className="popup-hint">
            This is a <strong>server</strong> path. Browsers can&apos;t browse your server&apos;s filesystem,
            but you can type or paste it here.
          </div>

          <div className="popup-section">
            <label className="popup-label">Training Device</label>
            <select
              className="popup-input"
              value={device}
              onChange={(e) => setDevice(e.target.value)}
            >
              <option value="cuda">GPU</option>
              <option value="cpu">CPU</option>
            </select>
          </div>

          <div className="popup-grid">
            <div>
              <label className="popup-label">Learning Rate</label>
              <input className="popup-input" type="number" step="0.0001" value={hyperparams.learning_rate} onChange={(e) => updateHyperparam("learning_rate", e.target.value)} />
            </div>
            <div>
              <label className="popup-label">LR Strategy</label>
              <select className="popup-input" value={hyperparams.lr_schedule} onChange={(e) => updateHyperparam("lr_schedule", e.target.value, String)}>
                <option value="constant">Constant</option>
                <option value="linear">Linear Decay</option>
                <option value="cosine">Cosine Annealing</option>
              </select>
            </div>
            <div>
              <label className="popup-label">Model Size</label>
              <select className="popup-input" value={hyperparams.model_size} onChange={(e) => updateHyperparam("model_size", e.target.value, String)}>
                <option value="small">Small</option>
                <option value="medium">Medium</option>
                <option value="large">Large</option>
              </select>
            </div>
            <div>
              <label className="popup-label">n_steps</label>
              <input className="popup-input" type="number" step="32" value={hyperparams.n_steps} onChange={(e) => updateHyperparam("n_steps", e.target.value)} />
            </div>
            <div>
              <label className="popup-label">Batch Size</label>
              <input className="popup-input" type="number" step="8" value={hyperparams.batch_size} onChange={(e) => updateHyperparam("batch_size", e.target.value)} />
            </div>
            <div>
              <label className="popup-label">Epochs</label>
              <input className="popup-input" type="number" step="1" value={hyperparams.n_epochs} onChange={(e) => updateHyperparam("n_epochs", e.target.value)} />
            </div>
            <div>
              <label className="popup-label">Gamma</label>
              <input className="popup-input" type="number" step="0.001" value={hyperparams.gamma} onChange={(e) => updateHyperparam("gamma", e.target.value)} />
            </div>
            <div>
              <label className="popup-label">GAE Lambda</label>
              <input className="popup-input" type="number" step="0.001" value={hyperparams.gae_lambda} onChange={(e) => updateHyperparam("gae_lambda", e.target.value)} />
            </div>
            <div>
              <label className="popup-label">Clip Range</label>
              <input className="popup-input" type="number" step="0.01" value={hyperparams.clip_range} onChange={(e) => updateHyperparam("clip_range", e.target.value)} />
            </div>
            <div>
              <label className="popup-label">Entropy Coef</label>
              <input className="popup-input" type="number" step="0.001" value={hyperparams.ent_coef} onChange={(e) => updateHyperparam("ent_coef", e.target.value)} />
            </div>
            <div>
              <label className="popup-label">Value Coef</label>
              <input className="popup-input" type="number" step="0.01" value={hyperparams.vf_coef} onChange={(e) => updateHyperparam("vf_coef", e.target.value)} />
            </div>
            <div>
              <label className="popup-label">Max Grad Norm</label>
              <input className="popup-input" type="number" step="0.1" value={hyperparams.max_grad_norm} onChange={(e) => updateHyperparam("max_grad_norm", e.target.value)} />
            </div>
          </div>
        </div>

        <div className="popup-actions">
          <button className="btn secondary" onClick={onClose}>Cancel (Esc)</button>
          <button className="btn primary" onClick={() => onConfirm?.(path, device, trainSteps, hyperparams)}>Save (Enter)</button>
        </div>
      </div>
    </div>,
    document.body
  );
};

export default SetPathPopup;
