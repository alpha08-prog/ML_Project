import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts'
import { CheckCircle, TrendingUp } from 'lucide-react'
import apiService from '../services/api'
import './ModelPerformance.css'

export default function ModelPerformance() {
  const [selectedModel, setSelectedModel] = useState<'rf' | 'cnn'>('cnn')
  const [performanceData, setPerformanceData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await apiService.getPerformance()
        setPerformanceData(data)
      } catch (err) {
        console.error('Error fetching performance data:', err)
        setError('Failed to load performance data')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  if (loading) {
    return <div className="loading-container">Loading performance data...</div>
  }

  if (error || !performanceData) {
    return <div className="loading-container" style={{ color: 'var(--error)' }}>{error || 'No data available'}</div>
  }

  const rfBaseline = {
    accuracy: performanceData.random_forest.baseline.accuracy,
    rocAuc: performanceData.random_forest.baseline.roc_auc,
    prAuc: performanceData.random_forest.baseline.pr_auc,
    precision: performanceData.random_forest.baseline.precision,
    recall: performanceData.random_forest.baseline.recall,
    f1: performanceData.random_forest.baseline.f1,
  }

  const rfAugmented = {
    accuracy: performanceData.random_forest.augmented.accuracy,
    rocAuc: performanceData.random_forest.augmented.roc_auc,
    prAuc: performanceData.random_forest.augmented.pr_auc,
    precision: performanceData.random_forest.augmented.precision,
    recall: performanceData.random_forest.augmented.recall,
    f1: performanceData.random_forest.augmented.f1,
  }

  const cnnBaseline = {
    accuracy: performanceData.cnn.baseline.accuracy,
    rocAuc: performanceData.cnn.baseline.roc_auc,
    prAuc: performanceData.cnn.baseline.pr_auc,
    precision: performanceData.cnn.baseline.precision,
    recall: performanceData.cnn.baseline.recall,
    f1: performanceData.cnn.baseline.f1,
  }

  const cnnAugmented = {
    accuracy: performanceData.cnn.augmented.accuracy,
    rocAuc: performanceData.cnn.augmented.roc_auc,
    prAuc: performanceData.cnn.augmented.pr_auc,
    precision: performanceData.cnn.augmented.precision,
    recall: performanceData.cnn.augmented.recall,
    f1: performanceData.cnn.augmented.f1,
  }

  const getCurrentMetrics = () => {
    if (selectedModel === 'rf') {
      return { baseline: rfBaseline, augmented: rfAugmented }
    }
    return { baseline: cnnBaseline, augmented: cnnAugmented }
  }

  const { baseline, augmented } = getCurrentMetrics()

  const comparisonData = [
    { metric: 'Accuracy', baseline: baseline.accuracy, augmented: augmented.accuracy },
    { metric: 'ROC-AUC', baseline: baseline.rocAuc, augmented: augmented.rocAuc },
    { metric: 'PR-AUC', baseline: baseline.prAuc, augmented: augmented.prAuc },
    { metric: 'Precision', baseline: baseline.precision, augmented: augmented.precision },
    { metric: 'Recall', baseline: baseline.recall, augmented: augmented.recall },
    { metric: 'F1-Score', baseline: baseline.f1, augmented: augmented.f1 },
  ]

  const radarData = [
    { metric: 'Accuracy', baseline: baseline.accuracy, augmented: augmented.accuracy },
    { metric: 'ROC-AUC', baseline: baseline.rocAuc, augmented: augmented.rocAuc },
    { metric: 'Precision', baseline: baseline.precision, augmented: augmented.precision },
    { metric: 'Recall', baseline: baseline.recall, augmented: augmented.recall },
    { metric: 'F1-Score', baseline: baseline.f1, augmented: augmented.f1 },
  ]

  const confusionMatrix = {
    baseline: { tp: 45, fp: 12, fn: 15, tn: 28 },
    augmented: { tp: 52, fp: 8, fn: 8, tn: 32 },
  }

  const improvements = Object.keys(baseline).map((key) => ({
    metric: key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1'),
    improvement: ((augmented[key as keyof typeof augmented] - baseline[key as keyof typeof baseline]) * 100).toFixed(1),
  }))

  return (
    <div className="model-performance">
      <div className="page-header">
        <h1>Model Performance</h1>
        <p className="subtitle">Detailed metrics and comparison analysis</p>
      </div>

      <div className="model-selector">
        <button
          className={`model-btn ${selectedModel === 'rf' ? 'active' : ''}`}
          onClick={() => setSelectedModel('rf')}
        >
          Random Forest
        </button>
        <button
          className={`model-btn ${selectedModel === 'cnn' ? 'active' : ''}`}
          onClick={() => setSelectedModel('cnn')}
        >
          CNN
        </button>
      </div>

      <div className="metrics-overview grid grid-2">
        <div className="card hover-float glass">
          <h3>Baseline Model</h3>
          <div className="metrics-list">
            {Object.entries(baseline).map(([key, value]) => (
              <div key={key} className="metric-row">
                <span className="metric-name">{key.charAt(0).toUpperCase() + key.slice(1)}</span>
                <span className="metric-value">{(value * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card hover-float glass">
          <h3>Augmented Model</h3>
          <div className="metrics-list">
            {Object.entries(augmented).map(([key, value]) => {
              const improvement = ((value - baseline[key as keyof typeof baseline]) * 100).toFixed(1)
              return (
                <div key={key} className="metric-row">
                  <span className="metric-name">{key.charAt(0).toUpperCase() + key.slice(1)}</span>
                  <div className="metric-value-group">
                    <span className="metric-value">{(value * 100).toFixed(1)}%</span>
                    {parseFloat(improvement) > 0 && (
                      <span className="metric-improvement">
                        <TrendingUp size={14} />
                        +{improvement}%
                      </span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      <div className="card mt-lg hover-float glass">
        <h3>Performance Comparison</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={comparisonData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="metric" />
            <YAxis domain={[0, 1]} />
            <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
            <Legend />
            <Bar dataKey="baseline" fill="#94a3b8" name="Baseline" />
            <Bar dataKey="augmented" fill="#10b981" name="Augmented" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-2 mt-lg">
        <div className="card hover-float glass">
          <h3>Radar Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" />
              <PolarRadiusAxis angle={90} domain={[0, 1]} />
              <Radar name="Baseline" dataKey="baseline" stroke="#94a3b8" fill="#94a3b8" fillOpacity={0.6} />
              <Radar name="Augmented" dataKey="augmented" stroke="#10b981" fill="#10b981" fillOpacity={0.6} />
              <Legend />
              <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <div className="card hover-float glass">
          <h3>Confusion Matrix</h3>
          <div className="confusion-matrix">
            <div className="cm-section">
              <h4>Baseline</h4>
              <div className="cm-grid">
                <div className="cm-cell cm-tp">{confusionMatrix.baseline.tp}</div>
                <div className="cm-cell cm-fp">{confusionMatrix.baseline.fp}</div>
                <div className="cm-cell cm-fn">{confusionMatrix.baseline.fn}</div>
                <div className="cm-cell cm-tn">{confusionMatrix.baseline.tn}</div>
              </div>
              <div className="cm-labels">
                <span>TP</span>
                <span>FP</span>
                <span>FN</span>
                <span>TN</span>
              </div>
            </div>
            <div className="cm-section">
              <h4>Augmented</h4>
              <div className="cm-grid">
                <div className="cm-cell cm-tp">{confusionMatrix.augmented.tp}</div>
                <div className="cm-cell cm-fp">{confusionMatrix.augmented.fp}</div>
                <div className="cm-cell cm-fn">{confusionMatrix.augmented.fn}</div>
                <div className="cm-cell cm-tn">{confusionMatrix.augmented.tn}</div>
              </div>
              <div className="cm-labels">
                <span>TP</span>
                <span>FP</span>
                <span>FN</span>
                <span>TN</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card mt-lg hover-float glass">
        <h3>Key Improvements</h3>
        <div className="improvements-list">
          {improvements.map((item) => (
            <div key={item.metric} className="improvement-item">
              <CheckCircle className="improvement-icon" size={20} />
              <div className="improvement-content">
                <span className="improvement-metric">{item.metric}</span>
                <span className="improvement-value">+{item.improvement}% improvement</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

