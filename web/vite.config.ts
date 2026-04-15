import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages serves this under adityachaturvedii.github.io/cardiocam/ so
// all asset URLs must be prefixed. Vite writes this prefix into the built
// index.html + asset tags. For local dev (base '/') Vite ignores the prefix.
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/cardiocam/' : '/',
  plugins: [react()],
}))
