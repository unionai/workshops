import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// VITE_API_BASE is injected at container start by start.sh, which gets it from the
// app environment's env vars (deploy.py wires it to the API app's endpoint).
export default defineConfig({
  plugins: [vue()],
  preview: {
    port: 8080,
    host: true,
    // Union serves apps under *.unionai.cloud; vite's preview server rejects
    // unknown hosts unless they are allow-listed.
    allowedHosts: ['.unionai.cloud', '.union.ai'],
  },
  server: {
    port: 5173,
    proxy: {
      // Local development: `npm run dev` against a locally running API.
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
