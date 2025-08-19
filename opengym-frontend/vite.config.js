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
    '/models' : 'http://localhost:8000'
    }
  }
}
);
