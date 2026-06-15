import React, { useState, useEffect } from "react";
import { createPortal } from "react-dom";

const SaveRolloutPopup = ({
  runId,
  isOpen,
  onConfirm,
  onClose,
  onSkip = null,
  title = "Save Rollouts",
  message = "",
  confirmLabel = "Save",
  skipLabel = "Continue Without Saving",
  showSkip = false,
}) => {
  const [filename, setFilename] = useState("rollouts_.json");

  // Reset default when reopened
  useEffect(() => {
    if (isOpen) {
      setFilename("rollouts_.json");
    }
  }, [isOpen]);

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

  if (!isOpen) return null;

  const handleConfirm = () => {
    const cleanName = filename.trim();
    if (!cleanName) {
      alert("Please enter a filename.");
      return;
    }
    onConfirm(cleanName); // 👈 pass back the filename
  };

  return createPortal(
    <div className="popup-backdrop" onClick={onClose}>
      <div
        className="popup-card"
        style={{ width: "min(560px, 92vw)" }}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="popup-header">
          <div>{title}</div>
          <button className="popup-close" onClick={onClose} aria-label="Close">x</button>
        </div>
        <div className="popup-body">
          {message ? (
            <div style={{ marginBottom: "0.9rem", color: "#475569", fontSize: "0.92rem", lineHeight: 1.45 }}>
              {message}
            </div>
          ) : null}

          <label htmlFor="filename" className="popup-label">
            Filename
          </label>
          <input
            id="filename"
            className="popup-input"
            type="text"
            value={filename}
            onChange={(e) => setFilename(e.target.value)}
          />
        </div>
        <div className="popup-actions" style={{ justifyContent: "center", flexWrap: "wrap" }}>
          <button className="btn secondary" onClick={onClose}>Cancel</button>
          {showSkip && onSkip ? (
            <button
              onClick={onSkip}
              style={{
                padding: "9px 13px",
                borderRadius: "12px",
                border: "1px solid #cbd5e1",
                backgroundColor: "white",
                color: "#334155",
                fontWeight: 700,
                cursor: "pointer",
              }}
            >
              {skipLabel}
            </button>
          ) : null}
          <button className="btn primary" onClick={handleConfirm}>{confirmLabel}</button>
        </div>
      </div>
    </div>,
    document.body
  );
};

export default SaveRolloutPopup;
