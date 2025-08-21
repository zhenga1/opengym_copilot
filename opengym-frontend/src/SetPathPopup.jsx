import React, { useEffect, useRef, useState } from "react";

const SetPathPopup = ({
  isOpen,
  defaultPath = "./models/ppo_model",
  onConfirm,
  onClose,
}) => {
  const [path, setPath] = useState(defaultPath);
  const [device, setDevice] = useState("cuda"); // default gpu = cuda
  const inputRef = useRef(null);
  const frozenDefaultRef = useRef(null);

  // reset path when opened & focus the input
  useEffect(() => {
    if (isOpen) {
      frozenDefaultRef.current = defaultPath; // store the initial value
      setPath(defaultPath);
      // focus after mount
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [isOpen, defaultPath]);

  // keyboard: Enter confirm, Esc close
  const onKeyDown = (e) => {
    if (e.key === "Enter") onConfirm?.(path, device);
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
        </div>

        <div className="popup-actions">
          <button className="btn secondary" onClick={onClose}>Cancel (Esc)</button>
          <button className="btn primary" onClick={() => onConfirm?.(path, device)}>Save (Enter)</button>
        </div>
      </div>
    </div>
  );
};

export default SetPathPopup;
