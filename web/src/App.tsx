import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Landing from './pages/Landing'
import Reader from './pages/Reader'

// GitHub Pages serves the app at /cardiocam/, so the router basename must
// match. import.meta.env.BASE_URL is '/cardiocam/' in production builds
// (set by vite.config.ts) and '/' during `npm run dev`.
const basename = import.meta.env.BASE_URL.replace(/\/$/, '') || '/'

export default function App() {
  return (
    <BrowserRouter basename={basename}>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/measure" element={<Reader />} />
      </Routes>
    </BrowserRouter>
  )
}
