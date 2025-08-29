// ✅ Valid in .jsx files
import { useState, useEffect } from "react";
import axios from "axios";


export default function ProgressBar({ isTraining, runId }) {
  const [progress, setProgress] = useState(0);
  const [processComplete, setProcessComplete] = useState(false);

  useEffect(() => {
    if (!isTraining) {
      setProgress(0);
      return;
    }
    if (!runId) {
      console.warn("Run ID not set yet, cannot pause/resume");
      setProgress(0);
      return;
    }

    const interval = setInterval(async () => {
      try {
        console.log("BEGIN GETTING from /progress");
        const res = await axios.get(`/progress/${runId}`);
        //console.log("RESPONSE html:", res.data);
        console.log("RESPONSE data:", res.data.progress);
        const value = res.data.progress;
        setProgress(value);

        if (value >= 100) {
          clearInterval(interval);
          setProcessComplete(true);
        }
      } catch (err) {
        console.error("Error fetching progress:", err);
        clearInterval(interval);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [isTraining]);

  if (!isTraining) return null;

  return (
    <div style={{ margin: '1.5rem auto', textAlign: 'center'}}>
      <div style={{
        height: '8px',
        width: '60%',
        backgroundColor: '#e5e7eb',
        borderRadius: '999px',
        overflow: 'hidden',
        margin: '0 auto',
        position: 'relative'
      }}>
        <div style={{
          height: '100%',
          width: `${progress}%`,
          backgroundColor: '#3b82f6',
          animation: 'progress-slide 1.5s infinite ease-in-out'
        }} />
      </div>
      <p style={{ marginTop: '0.5rem', color: '#4b5563', fontWeight: 500 }}>
        {processComplete ? "Training complete!" : "Training in progress..."}
      </p>
    </div>
  );
}
