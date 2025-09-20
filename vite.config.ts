import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Check if running in a sandbox environment where proxy won't work
const isInSandbox = process.env.NODE_ENV === 'development' && 
  (process.env.STACKBLITZ || process.env.CODESANDBOX || process.env.GITPOD)

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 8080,
    // Only enable proxy in local development, not in sandbox environments
    proxy: isInSandbox ? undefined : {
      '/predict': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false,
      },
      '/static': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false,
      },
      '/health': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
})