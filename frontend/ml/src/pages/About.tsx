import { useEffect, useState } from 'react'
import { Brain, Target, Zap, BarChart3, Users, FileText } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card'
import apiService from '../services/api'

interface ClassDistribution {
  good: number
  bad: number
}

interface DashboardData {
  total_subjects: number
  class_distribution: ClassDistribution
}

interface ModelMetrics {
  accuracy: number
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

export default function About() {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null)
  const [performanceData, setPerformanceData] = useState<PerformanceData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [dash, perf] = await Promise.all([
          apiService.getDashboard(),
          apiService.getPerformance(),
        ])
        setDashboardData(dash)
        setPerformanceData(perf)
      } catch {
        setError('Failed to load dataset details. Make sure the backend is running.')
      } finally {
        setLoading(false)
      }
    }
    fetchAll()
  }, [])

  const FEATURES = [
    {
      icon: <Brain size={32} />,
      title: 'GAN-Based Data Augmentation',
      description: 'Uses Wasserstein GAN with gradient penalty to generate synthetic minority class samples, effectively addressing class imbalance.',
    },
    {
      icon: <Target size={32} />,
      title: 'Dual Model Architecture',
      description: 'Implements both Random Forest (feature-based) and CNN (deep learning) classifiers for comprehensive performance comparison.',
    },
    {
      icon: <Zap size={32} />,
      title: 'Enhanced Performance',
      description: 'Augmented models show significant improvements across accuracy, ROC-AUC, and F1-score compared to baseline models.',
    },
    {
      icon: <BarChart3 size={32} />,
      title: 'Comprehensive Analysis',
      description: 'Includes statistical validation (KS tests), UMAP visualization, and explainability with Integrated Gradients.',
    },
  ]

  const METHODOLOGY = [
    { step: '1', title: 'Data Preprocessing', description: 'EEG signals from subjects, combining baseline and task recordings. Windowing at 512 samples with a 256 step downsampled to 128 Hz.' },
    { step: '2', title: 'GAN Training', description: 'Train Wasserstein GAN on minority class samples. The Generator creates synthetic windows while the Critic ensures distribution matching.' },
    { step: '3', title: 'Data Augmentation', description: 'Generate synthetic minority samples and combine with original training data to produce a balanced dataset.' },
    { step: '4', title: 'Model Training', description: 'Train Random Forest and CNN classifiers on both baseline and augmented datasets, then evaluate on held-out test sets.' },
    { step: '5', title: 'Validation & Analysis', description: 'Apply statistical tests (KS), dimensionality reduction (UMAP), and explainability analysis to validate the augmented data quality.' },
  ]

  return (
    <div className="about">
      <div className="about-hero">
        <h1>EEG Mental Arithmetic Classification</h1>
        <p className="hero-subtitle">
          A machine learning project using GAN-based data augmentation to improve
          the classification of mental arithmetic task performance quality from EEG signals.
        </p>
      </div>

      <Card className="mt-xl">
        <CardHeader><CardTitle>Problem Statement</CardTitle></CardHeader>
        <CardContent>
          <p>
            This project addresses the challenge of classifying EEG signals into mental arithmetic task performance quality.
            Subjects are categorized into two groups:
          </p>
          <ul className="info-list mt-md">
            <li>
              <strong>Good Quality (Group G):</strong>
              {' '}{dashboardData?.class_distribution?.good ?? '—'} subjects
            </li>
            <li>
              <strong>Bad Quality (Group B):</strong>
              {' '}{dashboardData?.class_distribution?.bad ?? '—'} subjects
            </li>
          </ul>
          <p className="mt-md">
            The inherent class imbalance poses a challenge for traditional machine learning.
            By using GAN-based synthetic data generation, we augment the minority class and significantly improve model robustness.
          </p>
        </CardContent>
      </Card>

      <div className="mt-xl">
        <h2>Key Features</h2>
        <div className="grid grid-2 mt-lg">
          {FEATURES.map((feature, idx) => (
            <Card key={idx} className="feature-card">
              <div className="feature-icon">{feature.icon}</div>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </Card>
          ))}
        </div>
      </div>

      <div className="mt-xl">
        <h2>Methodology</h2>
        <div className="methodology-timeline mt-lg">
          {METHODOLOGY.map((item, idx) => (
            <div key={idx} className="methodology-step">
              <div className="step-number">{item.step}</div>
              <div className="step-content">
                <h3>{item.title}</h3>
                <p>{item.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-2 mt-xl">
        <Card>
            <CardHeader><CardTitle>Dataset Information</CardTitle></CardHeader>
            <CardContent>
                <div className="dataset-info">
                    <div className="info-item">
                        <Users size={20} />
                        <div>
                            <strong>Subjects:</strong>
                            {dashboardData?.total_subjects ?? '—'} (
                            {dashboardData?.class_distribution?.good ?? '—'} Good,
                            {' '}{dashboardData?.class_distribution?.bad ?? '—'} Bad)
                        </div>
                    </div>
                    <div className="info-item">
                        <FileText size={20} />
                        <div>
                            <strong>Channels:</strong> 23 (10-20 system)
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>

        <Card>
            <CardHeader><CardTitle>Results Summary (Live)</CardTitle></CardHeader>
            <CardContent>
                {loading ? <p>Loading live results…</p> : error ? <p className="text-error">{error}</p> : (
                    <div className="grid grid-2">
                        <div className="result-item">
                            <h3>Random Forest</h3>
                            <ul>
                                <li>Base: {((performanceData?.random_forest?.baseline?.accuracy ?? 0) * 100).toFixed(1)}%</li>
                                <li>Aug: {((performanceData?.random_forest?.augmented?.accuracy ?? 0) * 100).toFixed(1)}%</li>
                            </ul>
                        </div>
                        <div className="result-item">
                            <h3>CNN</h3>
                            <ul>
                                <li>Base: {((performanceData?.cnn?.baseline?.accuracy ?? 0) * 100).toFixed(1)}%</li>
                                <li>Aug: {((performanceData?.cnn?.augmented?.accuracy ?? 0) * 100).toFixed(1)}%</li>
                            </ul>
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
      </div>

      <Card className="mt-xl">
        <CardHeader><CardTitle>Technologies Used</CardTitle></CardHeader>
        <CardContent>
          <div className="tech-tags">
            {['Python', 'PyTorch', 'scikit-learn', 'MNE', 'PyEDFlib', 'React', 'TypeScript', 'Recharts', 'SHAP', 'UMAP'].map(tech => (
              <span key={tech} className="tech-tag">{tech}</span>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
