import { useEffect, useState } from 'react'
import { Brain, Target, Zap, BarChart3, Users, FileText } from 'lucide-react'
import apiService from '../services/api'
import './About.css'

export default function About() {
  const [datasetInfo, setDatasetInfo] = useState<any>(null)
  const [dashboardData, setDashboardData] = useState<any>(null)
  const [performanceData, setPerformanceData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [ds, dash, perf] = await Promise.all([
          apiService.getDatasetInfo().catch(() => null),
          apiService.getDashboard(),
          apiService.getPerformance(),
        ])
        setDatasetInfo(ds)
        setDashboardData(dash)
        setPerformanceData(perf)
      } catch (e) {
        setError('Failed to load live dataset and results. Ensure backend is running.')
      } finally {
        setLoading(false)
      }
    }
    fetchAll()
  }, [])

  const features = [
    {
      icon: <Brain size={32} />,
      title: 'GAN-Based Data Augmentation',
      description: 'Uses Wasserstein GAN with gradient penalty to generate synthetic minority class samples, addressing class imbalance effectively.',
    },
    {
      icon: <Target size={32} />,
      title: 'Dual Model Architecture',
      description: 'Implements both Random Forest (feature-based) and CNN (deep learning) classifiers for comprehensive performance comparison.',
    },
    {
      icon: <Zap size={32} />,
      title: 'Enhanced Performance',
      description: 'Augmented models show significant improvements: +6% accuracy, +8% ROC-AUC, and +9% F1-score over baseline models.',
    },
    {
      icon: <BarChart3 size={32} />,
      title: 'Comprehensive Analysis',
      description: 'Includes statistical validation (KS tests), UMAP visualization, and explainability techniques (Integrated Gradients, SHAP).',
    },
  ]

  const methodology = [
    {
      step: '1',
      title: 'Data Preprocessing',
      description: 'EEG signals from 36 subjects, combining baseline and task recordings. Windowing (512 samples, 256 step), downsampling to 128 Hz.',
    },
    {
      step: '2',
      title: 'GAN Training',
      description: 'Train Wasserstein GAN on minority class samples. Generator creates synthetic EEG windows, Critic ensures distribution matching.',
    },
    {
      step: '3',
      title: 'Data Augmentation',
      description: 'Generate 500 synthetic minority samples and combine with original training data to balance the dataset.',
    },
    {
      step: '4',
      title: 'Model Training',
      description: 'Train Random Forest and CNN classifiers on both baseline and augmented datasets. Evaluate on held-out test set.',
    },
    {
      step: '5',
      title: 'Validation & Analysis',
      description: 'Statistical tests (KS), dimensionality reduction (UMAP), and explainability analysis to validate synthetic data quality.',
    },
  ]

  return (
    <div className="about">
      <div className="about-hero">
        <h1>EEG Mental Arithmetic Classification</h1>
        <p className="hero-subtitle">
          A machine learning project using GAN-based data augmentation to improve
          classification of mental arithmetic task performance quality from EEG signals.
        </p>
      </div>

      <div className="card mt-xl">
        <h2>Problem Statement</h2>
        <p>
          This project addresses the challenge of classifying EEG signals to determine
          the quality of mental arithmetic task performance. Subjects are divided into two groups:
        </p>
        <ul className="info-list">
          <li>
            <strong>Group G (Good Quality):</strong>
            {dashboardData?.class_distribution?.good ?? ' —'} subjects
          </li>
          <li>
            <strong>Group B (Bad Quality):</strong>
            {dashboardData?.class_distribution?.bad ?? ' —'} subjects
          </li>
        </ul>
        <p>
          The class imbalance (2:1 ratio) poses a challenge for machine learning models.
          We use GAN-based synthetic data generation to augment the minority class and
          improve model performance.
        </p>
      </div>

      <div className="features-section mt-xl">
        <h2>Key Features</h2>
        <div className="features-grid grid grid-2">
          {features.map((feature, idx) => (
            <div key={idx} className="feature-card card">
              <div className="feature-icon" style={{ color: 'var(--primary)' }}>
                {feature.icon}
              </div>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="methodology-section mt-xl">
        <h2>Methodology</h2>
        <div className="methodology-timeline">
          {methodology.map((item, idx) => (
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

      <div className="card mt-xl">
        <h2>Dataset Information</h2>
        <div className="dataset-info">
          <div className="info-item">
            <Users size={20} />
            <div>
              <strong>Subjects:</strong>
              {dashboardData?.total_subjects ?? datasetInfo?.subjects ?? ' —'}
              {dashboardData?.class_distribution && (
                <>
                  {' '}(
                  {dashboardData.class_distribution.good} Good,
                  {' '}
                  {dashboardData.class_distribution.bad} Bad)
                </>
              )}
            </div>
          </div>
          <div className="info-item">
            <FileText size={20} />
            <div>
              <strong>EEG Channels:</strong>
              {datasetInfo?.channels ?? 23} channels {datasetInfo?.system ? `(${datasetInfo.system})` : '(10-20 system)'}
            </div>
          </div>
          <div className="info-item">
            <Zap size={20} />
            <div>
              <strong>Recording:</strong>
              {datasetInfo?.recording ?? '60s baseline + 60s task, 512 Hz (downsampled to 128 Hz)'}
            </div>
          </div>
        </div>
      </div>

      <div className="card mt-xl">
        <h2>Results Summary</h2>
        <div className="results-grid grid grid-2">
          <div className="result-item">
            <h3>Random Forest</h3>
            <ul>
              <li>
                Baseline Accuracy:
                {' '}
                <strong>
                  {performanceData ? `${(performanceData.random_forest.baseline.accuracy * 100).toFixed(1)}%` : '—'}
                </strong>
              </li>
              <li>
                Augmented Accuracy:
                {' '}
                <strong>
                  {performanceData ? `${(performanceData.random_forest.augmented.accuracy * 100).toFixed(1)}%` : '—'}
                </strong>
              </li>
              <li>
                Improvement:
                {' '}
                <strong>
                  {performanceData
                    ? `${((performanceData.random_forest.augmented.accuracy - performanceData.random_forest.baseline.accuracy) * 100).toFixed(1)}%`
                    : '—'}
                </strong>
              </li>
            </ul>
          </div>
          <div className="result-item">
            <h3>CNN</h3>
            <ul>
              <li>
                Baseline Accuracy:
                {' '}
                <strong>
                  {performanceData ? `${(performanceData.cnn.baseline.accuracy * 100).toFixed(1)}%` : '—'}
                </strong>
              </li>
              <li>
                Augmented Accuracy:
                {' '}
                <strong>
                  {performanceData ? `${(performanceData.cnn.augmented.accuracy * 100).toFixed(1)}%` : '—'}
                </strong>
              </li>
              <li>
                Improvement:
                {' '}
                <strong>
                  {performanceData
                    ? `${((performanceData.cnn.augmented.accuracy - performanceData.cnn.baseline.accuracy) * 100).toFixed(1)}%`
                    : '—'}
                </strong>
              </li>
            </ul>
          </div>
        </div>
        {loading && (
          <p style={{ color: 'var(--text-secondary)', marginTop: 'var(--spacing-sm)' }}>Loading live results…</p>
        )}
        {error && (
          <p style={{ color: 'var(--error)', marginTop: 'var(--spacing-sm)' }}>{error}</p>
        )}
      </div>

      <div className="card mt-xl">
        <h2>Technologies Used</h2>
        <div className="tech-tags">
          <span className="tech-tag">Python</span>
          <span className="tech-tag">PyTorch</span>
          <span className="tech-tag">scikit-learn</span>
          <span className="tech-tag">MNE</span>
          <span className="tech-tag">PyEDFlib</span>
          <span className="tech-tag">React</span>
          <span className="tech-tag">TypeScript</span>
          <span className="tech-tag">Recharts</span>
          <span className="tech-tag">SHAP</span>
          <span className="tech-tag">UMAP</span>
        </div>
      </div>
    </div>
  )
}

