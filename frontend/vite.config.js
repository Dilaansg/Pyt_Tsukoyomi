import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    proxy: {
      '/simular-friccion': 'http://127.0.0.1:8000',
      '/feedback': 'http://127.0.0.1:8000',
      '/detectar-contexto-visual': 'http://127.0.0.1:8000',
    }
  }
})
