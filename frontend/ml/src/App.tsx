import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import ModelPerformance from './pages/ModelPerformance'
import DataVisualization from './pages/DataVisualization'
import Predictions from './pages/Predictions'
import About from './pages/About'
import './App.css'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/performance" element={<ModelPerformance />} />
          <Route path="/visualization" element={<DataVisualization />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App
