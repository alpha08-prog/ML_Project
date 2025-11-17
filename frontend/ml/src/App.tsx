import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import ModelPerformance from './pages/ModelPerformance'
import DataVisualization from './pages/DataVisualization'
import Predictions from './pages/Predictions'
import About from './pages/About'
import './App.css'
import { AnimatePresence, motion } from 'framer-motion'
import { useLocation } from 'react-router-dom'

function App() {
  return (
    <Router>
      <Layout>
        <RouteTransitionContainer />
      </Layout>
    </Router>
  )
}

function RouteTransitionContainer() {
  const location = useLocation()
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.2, ease: 'easeOut' }}
      >
        <Routes location={location}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/performance" element={<ModelPerformance />} />
          <Route path="/visualization" element={<DataVisualization />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </motion.div>
    </AnimatePresence>
  )
}

export default App
