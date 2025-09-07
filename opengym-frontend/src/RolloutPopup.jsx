import React, { useState, useEffect } from "react";

const SaveRolloutPopup = ({ runId, isOpen, onConfirm, onClose }) => {
  const [filename, setFilename] = useState("rollouts_.json");

  // Reset default when reopened
  useEffect(() => {
    if (isOpen) {
      setFilename("rollouts_.json");
    }
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

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "rgba(0,0,0,0.4)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
      }}
    >
      <div
        style={{
          backgroundColor: "white",
          padding: "1.5rem",
          borderRadius: "8px",
          minWidth: "320px",
          textAlign: "center",
          boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
        }}
      >
        <h3 style={{ marginBottom: "1rem" }}>Save Rollouts</h3>

        <label htmlFor="filename" style={{ fontWeight: 600 }}>
          Filename:
        </label>
        <br />
        <input
          id="filename"
          type="text"
          value={filename}
          onChange={(e) => setFilename(e.target.value)}
          style={{
            width: "90%",
            padding: "6px",
            margin: "0.75rem 0",
            border: "1px solid #d1d5db",
            borderRadius: "4px",
          }}
        />

        <div style={{ display: "flex", justifyContent: "center", gap: "1rem" }}>
          <button
            onClick={onClose}
            style={{
              padding: "6px 12px",
              borderRadius: "4px",
              border: "1px solid #ccc",
              backgroundColor: "#f3f4f6",
              cursor: "pointer",
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            style={{
              padding: "6px 12px",
              borderRadius: "4px",
              border: "none",
              backgroundColor: "#2563eb",
              color: "white",
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
};

export default SaveRolloutPopup;
