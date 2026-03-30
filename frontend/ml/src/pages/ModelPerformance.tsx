import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar
} from 'recharts'
import { CheckCircle} from 'lucide-react'
import apiService from '../services/api'
import './ModelPerformance.css'

interface ModelMetrics {
  accuracy: number
  roc_auc: number
  pr_auc: number
  precision: number
  recall: number
  f1: number
}

interface PerformanceData {
  random_forest: {
    baseline: ModelMetrics
    augmented: ModelMetrics
  }
  cnn?: {
    baseline: ModelMetrics
    augmented: ModelMetrics
  }
}

export default function ModelPerformance() {
  const [selectedModel, setSelectedModel] = useState<'rf' | 'cnn'>('rf')
  const [performanceData, setPerformanceData] = useState<PerformanceData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await apiService.getPerformance()
        setPerformanceData(data)
      } catch (err) {
        console.error(err)
        setError('Failed to load performance data')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  if (loading) return <div className="loading-container">Loading...</div>
  if (error || !performanceData) {
    return <div className="loading-container">{error || 'No data'}</div>
  }

  const hasCNN = !!performanceData.cnn

  // RF
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

  // CNN (safe)
  const cnnBaseline = hasCNN ? {
    accuracy: performanceData.cnn!.baseline.accuracy,
    rocAuc: performanceData.cnn!.baseline.roc_auc,
    prAuc: performanceData.cnn!.baseline.pr_auc,
    precision: performanceData.cnn!.baseline.precision,
    recall: performanceData.cnn!.baseline.recall,
    f1: performanceData.cnn!.baseline.f1,
  } : null

  const cnnAugmented = hasCNN ? {
    accuracy: performanceData.cnn!.augmented.accuracy,
    rocAuc: performanceData.cnn!.augmented.roc_auc,
    prAuc: performanceData.cnn!.augmented.pr_auc,
    precision: performanceData.cnn!.augmented.precision,
    recall: performanceData.cnn!.augmented.recall,
    f1: performanceData.cnn!.augmented.f1,
  } : null

  const getCurrentMetrics = () => {
    if (selectedModel === 'cnn' && hasCNN && cnnBaseline && cnnAugmented) {
      return { baseline: cnnBaseline, augmented: cnnAugmented }
    }
    return { baseline: rfBaseline, augmented: rfAugmented }
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

  const radarData = comparisonData

  const improvements = Object.keys(baseline).map((key) => ({
    metric: key,
    improvement: (
      (augmented[key as keyof typeof augmented] -
        baseline[key as keyof typeof baseline]) * 100
    ).toFixed(1),
  }))

  return (
    <div className="model-performance">

      <h1>Model Performance</h1>

      <div className="model-selector">
        <button
          className={selectedModel === 'rf' ? 'active' : ''}
          onClick={() => setSelectedModel('rf')}
        >
          Random Forest
        </button>

        <button
          className={selectedModel === 'cnn' ? 'active' : ''}
          onClick={() => hasCNN && setSelectedModel('cnn')}
          disabled={!hasCNN}
          style={{ opacity: hasCNN ? 1 : 0.5 }}
        >
          CNN {!hasCNN && '(Not Available)'}
        </button>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={comparisonData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="metric" />
          <YAxis domain={[0, 1]} />
          <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
          <Legend />
          <Bar dataKey="baseline" fill="#94a3b8" />
          <Bar dataKey="augmented" fill="#10b981" />
        </BarChart>
      </ResponsiveContainer>

      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={radarData}>
          <PolarGrid />
          <PolarAngleAxis dataKey="metric" />
          <PolarRadiusAxis domain={[0, 1]} />
          <Radar name="Baseline" dataKey="baseline" stroke="#94a3b8" fillOpacity={0.6} />
          <Radar name="Augmented" dataKey="augmented" stroke="#10b981" fillOpacity={0.6} />
        </RadarChart>
      </ResponsiveContainer>

      <div>
        {improvements.map((item) => (
          <div key={item.metric}>
            <CheckCircle size={16} /> {item.metric}: +{item.improvement}%
          </div>
        ))}
      </div>

    </div>
  )
}