import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server:{
    proxy: {
    '/progress': 'http://localhost:8000',
    '/start': 'http://localhost:8000',
    '/pause_rollout': 'http://localhost:8000',
    '/load_model': 'http://localhost:8000',
    '/upload_model': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      secure : false,
    },
    '/models' : 'http://localhost:8000',
    '/get_model_path': 'http://localhost:8000',
    '/set_training_dir': 'http://localhost:8000',
    '/unique_run_id': 'http://localhost:8000',
    "/training_runs": {
        target: "http://localhost:8000", // FastAPI backend
        changeOrigin: true,
        secure: false,
      },
    "/rollout_speed": 'http://localhost:8000',
    "/save_rollouts_data": 'http://localhost:8000',
    '/change_number_of_steps': 'http://localhost:8000',
    }
  }
}
);
