import { Line } from "react-chartjs-2";

export default function RolloutCard({ frames = [], currentFrame = 0, episodeInfo = { episode: 0, reward: 0 }, rollouts = [] }) {
  return (
    <div>
      {/* Frame display */}
      {frames.length > 0 && (
        <div style={{ marginTop: '2rem', textAlign: 'center' }}>
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

      {/* Reward Chart */}
      <div style={{ marginTop: '3rem' }}>
        <h3 style={{ fontSize: '1.25rem', color: '#6366f1' }}>📈 Reward Chart</h3>
        <p>
          Episode <strong>{episodeInfo.episode}</strong>, Reward:{' '}
          <strong style={{ color: '#10b981' }}>{episodeInfo.reward}</strong>
        </p>
        <div style={{ width: '100%', maxWidth: '600px', height: '300px' }}>
          <Line
            data={{
              labels: rollouts.map((r) => r.episode).reverse(),
              datasets: [
                {
                  label: "Reward",
                  data: rollouts.map((r) => r.reward).reverse(),
                  fill: false,
                  borderColor: 'rgb(56, 189, 248)',
                  backgroundColor: 'rgba(56, 189, 248, 0.2)',
                  tension: 0.25,
                },
              ],
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                x: { title: { display: true, text: "Episode" } },
                y: { title: { display: true, text: "Reward" } },
              },
            }}
          />
        </div>
      </div>
    </div>
  );
}
