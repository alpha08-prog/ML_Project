import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts'
import { Eye, Download, Maximize2 } from 'lucide-react'
import apiService from '../services/api'
import './DataVisualization.css'

// Simple synthetic signal generator for demo view
function generateEEGSignal(samples: number, seed: number) {
  const data: { time: number; amplitude: number }[] = []
  const freq = 8 + (seed % 5) // 8-12 Hz alpha-like
  const noiseLevel = 0.2
  for (let t = 0; t < samples; t++) {
    const signal = Math.sin((2 * Math.PI * freq * t) / samples) + (Math.cos((2 * Math.PI * (freq / 2) * t) / samples) * 0.5)
    const noise = (Math.random() - 0.5) * noiseLevel
    data.push({ time: t, amplitude: signal + noise })
  }
  return data
}

export default function DataVisualization() {
  const [selectedChannel, setSelectedChannel] = useState(0)
  const [selectedView, setSelectedView] = useState<'eeg' | 'umap' | 'distribution'>('eeg')
  const [eegData, setEegData] = useState<any[]>([])
  const [umapData, setUmapData] = useState<any[]>([])
  const [channelImportance, setChannelImportance] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch EEG signal data
  useEffect(() => {
    const fetchEEGSignal = async () => {
      try {
        setLoading(true)
        const data = await apiService.getEEGSignal(selectedChannel)
        setEegData(data.time.map((t: number, i: number) => ({
          time: t,
          amplitude: data.amplitude[i]
        })))
      } catch (err) {
        console.error('Error fetching EEG signal:', err)
        setError('Failed to load EEG signal')
      } finally {
        setLoading(false)
      }
    }
    if (selectedView === 'eeg') {
      fetchEEGSignal()
    }
  }, [selectedChannel, selectedView])

  // Fetch UMAP data
  useEffect(() => {
    const fetchUMAP = async () => {
      try {
        setLoading(true)
        const data = await apiService.getUMAP()
        setUmapData(data.points || [])
      } catch (err) {
        console.error('Error fetching UMAP data:', err)
        setError('Failed to load UMAP data')
      } finally {
        setLoading(false)
      }
    }
    if (selectedView === 'umap') {
      fetchUMAP()
    }
  }, [selectedView])

  // Fetch channel importance
  useEffect(() => {
    const fetchChannelImportance = async () => {
      try {
        setLoading(true)
        const data = await apiService.getChannelImportance()
        setChannelImportance(data)
      } catch (err) {
        console.error('Error fetching channel importance:', err)
        setError('Failed to load channel importance')
      } finally {
        setLoading(false)
      }
    }
    if (selectedView === 'distribution') {
      fetchChannelImportance()
    }
  }, [selectedView])


  const channelNames = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'A1', 'A2', 'Fpz', 'Oz'
  ]

  return (
    <div className="data-visualization">
      <div className="page-header">
        <h1>Data Visualization</h1>
        <p className="subtitle">Explore EEG signals, UMAP embeddings, and feature distributions</p>
      </div>

      <div className="view-selector">
        <button
          className={`view-btn ${selectedView === 'eeg' ? 'active' : ''}`}
          onClick={() => setSelectedView('eeg')}
        >
          <Eye size={18} />
          EEG Signals
        </button>
        <button
          className={`view-btn ${selectedView === 'umap' ? 'active' : ''}`}
          onClick={() => setSelectedView('umap')}
        >
          <Maximize2 size={18} />
          UMAP Embedding
        </button>
        <button
          className={`view-btn ${selectedView === 'distribution' ? 'active' : ''}`}
          onClick={() => setSelectedView('distribution')}
        >
          <Download size={18} />
          Channel Analysis
        </button>
      </div>

      {selectedView === 'eeg' && (
        <div className="visualization-section">
          <div className="card hover-float glass">
            <div className="chart-header">
              <h3>EEG Signal - Channel {channelNames[selectedChannel] || `Ch${selectedChannel}`}</h3>
              <div className="channel-selector">
                <label>Select Channel:</label>
                <select
                  value={selectedChannel}
                  onChange={(e) => setSelectedChannel(Number(e.target.value))}
                  className="channel-select"
                >
                  {channelNames.map((name, idx) => (
                    <option key={idx} value={idx}>
                      {name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={eegData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" label={{ value: 'Time (samples)', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Amplitude (μV)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Line type="monotone" dataKey="amplitude" stroke="#6366f1" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
            <div className="chart-info">
              <p>Baseline recording (60s) + Task recording (60s) | Downsampled to 128 Hz</p>
            </div>
          </div>

          <div className="card mt-lg hover-float glass">
            <h3>Multi-Channel Overview</h3>
            <ResponsiveContainer width="100%" height={500}>
              <LineChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                {channelNames.slice(0, 8).map((name, idx) => (
                  <Line
                    key={idx}
                    type="monotone"
                    dataKey={`ch${idx}`}
                    data={generateEEGSignal(128, idx).map((d) => ({ ...d, [`ch${idx}`]: d.amplitude }))}
                    stroke={`hsl(${idx * 45}, 70%, 50%)`}
                    strokeWidth={1.5}
                    name={name}
                    dot={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {selectedView === 'umap' && (
        <div className="visualization-section">
          <div className="card hover-float glass">
            <h3>UMAP Embedding Visualization</h3>
            <p className="chart-description">
              Dimensionality reduction showing the distribution of real minority samples,
              synthetic GAN-generated samples, and majority class samples in 2D space.
            </p>
            {loading ? (
              <div className="loading-container">Loading UMAP data...</div>
            ) : error ? (
              <div className="loading-container" style={{ color: 'var(--error)' }}>{error}</div>
            ) : (
              <ResponsiveContainer width="100%" height={500}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="x" label={{ value: 'UMAP Dimension 1', position: 'insideBottom', offset: -5 }} />
                  <YAxis dataKey="y" label={{ value: 'UMAP Dimension 2', angle: -90, position: 'insideLeft' }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Legend />
                  <Scatter name="Real Minority" data={umapData.filter(d => d.type === 'real')} fill="#10b981" />
                  <Scatter name="Synthetic" data={umapData.filter(d => d.type === 'synthetic')} fill="#6366f1" />
                  <Scatter name="Majority" data={umapData.filter(d => d.type === 'majority')} fill="#f59e0b" />
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      )}

      {selectedView === 'distribution' && (
        <div className="visualization-section">
          <div className="card hover-float glass">
            <h3>Channel Importance Analysis</h3>
            <p className="chart-description">
              Integrated Gradients analysis showing feature importance per channel for
              real vs synthetic samples.
            </p>
            {loading ? (
              <div className="loading-container">Loading channel importance data...</div>
            ) : error ? (
              <div className="loading-container" style={{ color: 'var(--error)' }}>{error}</div>
            ) : channelImportance ? (
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={channelImportance.channels.map((ch: string, i: number) => ({
                  channel: ch,
                  real: channelImportance.real[i],
                  synthetic: channelImportance.synthetic[i]
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="channel" />
                  <YAxis label={{ value: 'Mean |IG|', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="real" stroke="#10b981" name="Real Samples" strokeWidth={2} />
                  <Line type="monotone" dataKey="synthetic" stroke="#6366f1" name="Synthetic Samples" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            ) : null}
          </div>

          <div className="card mt-lg hover-float glass">
            <h3>KS Test Results</h3>
            <p className="chart-description">
              Kolmogorov-Smirnov test statistics comparing real minority vs synthetic
              sample distributions per channel.
            </p>
            {channelImportance ? (
              <div className="ks-results">
                {channelImportance.channels.slice(0, 10).map((ch: string, idx: number) => {
                  const importance = channelImportance.importance[idx]
                  return (
                    <div key={idx} className="ks-item">
                      <span className="ks-channel">{ch}</span>
                      <div className="ks-bar-container">
                        <div
                          className="ks-bar"
                          style={{ width: `${importance * 300}%`, background: importance > 0.2 ? '#ef4444' : '#10b981' }}
                        />
                      </div>
                      <span className="ks-value">{(importance * 100).toFixed(1)}%</span>
                    </div>
                  )
                })}
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  )
}

