import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // Proxy API calls to the FastAPI backend during development
    // so the browser never sees CORS issues on localhost.
    proxy: {
      '/api': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
    },
  },
})
