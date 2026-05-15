import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backendTarget = process.env.VITE_DEV_BACKEND_URL || 'http://localhost:8000';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server:{
    proxy: {
    '/progress': backendTarget,
    '/start': backendTarget,
    '/pause_rollout': backendTarget,
    '/load_model': backendTarget,
    '/upload_model': {
      target: backendTarget,
      changeOrigin: true,
      secure : false,
    },
    '/models' : backendTarget,
    '/get_model_path': backendTarget,
    '/set_training_dir': backendTarget,
    '/stop_training': backendTarget,
    '/unique_run_id': backendTarget,
    '/reward_config': backendTarget,
    '/propose_task_config': backendTarget,
    '/apply_task_config': backendTarget,
    '/task_config_status': backendTarget,
    "/training_runs": {
        target: backendTarget,
        changeOrigin: true,
        secure: false,
      },
    "/rollout_speed": backendTarget,
    "/save_rollouts_data": backendTarget,
    "/load_rollouts_data": backendTarget,
    "/rollouts_files": backendTarget,
    '/change_number_of_steps': backendTarget,
    '/delete_all_temp_models': backendTarget,
    '/healthz': backendTarget,
    '/ws': {
      target: backendTarget,
      ws: true,
      changeOrigin: true,
      secure: false,
    },
    }
  }
}
);
