// ✅ Valid in .jsx files
import { useState, useEffect } from "react";
import axios from "axios";


export default function ProgressBar({ isTraining }) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!isTraining) {
      setProgress(0);
      return;
    }

    const interval = setInterval(async () => {
      try {
        const res = await axios.get("/progress");
        const value = res.data.progress;
        setProgress(value);

        if (value >= 100) {
          clearInterval(interval);
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
    <div style={{ margin: '1.5rem auto', textAlign: 'center' }}>
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
        Training in progress...
      </p>
    </div>
  );
}
