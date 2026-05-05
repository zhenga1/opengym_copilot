import React, { useEffect, useRef, useState } from "react";

const SetPathPopup = ({
  isOpen,
  defaultPath = "./models/ppo_model",
  defaultHyperparams = {},
  onConfirm,
  onClose,
}) => {
  const [path, setPath] = useState(defaultPath);
  const [device, setDevice] = useState("cuda"); // default gpu = cuda
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
  const frozenDefaultRef = useRef(null);

  // reset path when opened & focus the input
  useEffect(() => {
    if (isOpen) {
      frozenDefaultRef.current = defaultPath; // store the initial value
      setPath(defaultPath);
      setHyperparams((prev) => ({ ...prev, ...defaultHyperparams }));
      // focus after mount
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [isOpen, defaultPath, defaultHyperparams]);

  const updateHyperparam = (key, value, cast = Number) => {
    setHyperparams((prev) => ({
      ...prev,
      [key]: cast === Number ? Number(value) : value,
    }));
  };

  // keyboard: Enter confirm, Esc close
  const onKeyDown = (e) => {
    if (e.key === "Enter") onConfirm?.(path, device, hyperparams);
    if (e.key === "Escape") onClose?.();
  };

  if (!isOpen) return null;

  return (
    <div className="popup-backdrop" onClick={onClose}>
      <div className="popup-card" onClick={(e) => e.stopPropagation()}>
        <div className="popup-header">
          <div>Set Training Path</div>
          <button className="popup-close" onClick={onClose} aria-label="Close">×</button>
        </div>

        <div className="popup-body">
          <label className="popup-label">Training folder (server-side path)</label>
          <input
            ref={inputRef}
            className="popup-input"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={defaultPath}
          />
          <div className="popup-hint">
            This is a <strong>server</strong> path. Browsers can’t browse your server’s filesystem,
            but you can type or paste it here.
          </div>

          <label style={{ display: "block", marginBottom: 6 }}>Training Device</label>
            <select
            value={device}
            onChange={(e) => setDevice(e.target.value === "gpu" ? "cuda" : "cpu")}
            style={{ width: "100%", marginBottom: 12 }}
            >
            <option value="gpu">GPU</option>
            <option value="cpu">CPU</option>
            </select>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 12 }}>
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
          <button className="btn primary" onClick={() => onConfirm?.(path, device, hyperparams)}>Save (Enter)</button>
        </div>
      </div>
    </div>
  );
};

export default SetPathPopup;
