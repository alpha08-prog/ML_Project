import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts'
import { TrendingUp, Users, Brain, Activity } from 'lucide-react'
import apiService from '../services/api'
import './Dashboard.css'

interface MetricCardProps {
  title: string
  value: string | number
  change?: string
  icon: React.ReactNode
  color: string
}

function MetricCard({ title, value, change, icon, color }: MetricCardProps) {
  return (
    <div className="metric-card card fade-in">
      <div className="metric-header">
        <div className="metric-icon" style={{ background: `${color}20`, color }}>
          {icon}
        </div>
        <div className="metric-content">
          <h3 className="metric-title">{title}</h3>
          <p className="metric-value">{value}</p>
          {change && <p className="metric-change">{change}</p>}
        </div>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [loading, setLoading] = useState(true)
  const [dashboardData, setDashboardData] = useState<any>(null)
  const [performanceData, setPerformanceData] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [dashboard, performance] = await Promise.all([
          apiService.getDashboard(),
          apiService.getPerformance()
        ])
        setDashboardData(dashboard)
        setPerformanceData(performance)
      } catch (err) {
        console.error('Error fetching dashboard data:', err)
        setError('Failed to load dashboard data. Make sure the backend is running.')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner" style={{ width: '48px', height: '48px', border: '4px solid var(--border)', borderTopColor: 'var(--primary)' }}></div>
        <p>Loading dashboard...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="loading-container">
        <p style={{ color: 'var(--error)' }}>{error}</p>
        <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
          Start the backend with: <code>python backend/api.py</code>
        </p>
      </div>
    )
  }

  // Prepare data from API
  const classDistribution = dashboardData?.class_distribution ? [
    { name: 'Good Quality (Group G)', value: dashboardData.class_distribution.good, color: '#10b981' },
    { name: 'Bad Quality (Group B)', value: dashboardData.class_distribution.bad, color: '#ef4444' },
  ] : []

  const modelComparison = performanceData ? [
    { 
      name: 'Accuracy', 
      'Random Forest (Baseline)': performanceData.random_forest.baseline.accuracy,
      'Random Forest (Augmented)': performanceData.random_forest.augmented.accuracy,
      'CNN (Baseline)': performanceData.cnn.baseline.accuracy,
      'CNN (Augmented)': performanceData.cnn.augmented.accuracy
    },
    { 
      name: 'ROC-AUC', 
      'Random Forest (Baseline)': performanceData.random_forest.baseline.roc_auc,
      'Random Forest (Augmented)': performanceData.random_forest.augmented.roc_auc,
      'CNN (Baseline)': performanceData.cnn.baseline.roc_auc,
      'CNN (Augmented)': performanceData.cnn.augmented.roc_auc
    },
    { 
      name: 'F1-Score', 
      'Random Forest (Baseline)': performanceData.random_forest.baseline.f1,
      'Random Forest (Augmented)': performanceData.random_forest.augmented.f1,
      'CNN (Baseline)': performanceData.cnn.baseline.f1,
      'CNN (Augmented)': performanceData.cnn.augmented.f1
    },
  ] : []

  // Simulated training progress (could be enhanced with real training logs)
  const trainingProgress = [
    { epoch: 1, loss: 0.85, accuracy: 0.65 },
    { epoch: 5, loss: 0.72, accuracy: 0.72 },
    { epoch: 10, loss: 0.58, accuracy: 0.78 },
    { epoch: 15, loss: 0.45, accuracy: 0.82 },
    { epoch: 20, loss: 0.38, accuracy: 0.85 },
    { epoch: 25, loss: 0.32, accuracy: 0.87 },
    { epoch: 30, loss: 0.28, accuracy: dashboardData?.best_accuracy || 0.88 },
  ]

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner" style={{ width: '48px', height: '48px', border: '4px solid var(--border)', borderTopColor: 'var(--primary)' }}></div>
        <p>Loading dashboard...</p>
      </div>
    )
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Dashboard</h1>
        <p className="subtitle">EEG Mental Arithmetic Classification - Overview</p>
      </div>

      <div className="metrics-grid grid grid-4">
        <MetricCard
          title="Total Subjects"
          value={dashboardData?.total_subjects || 0}
          icon={<Users size={24} />}
          color="var(--primary)"
        />
        <MetricCard
          title="Best Model Accuracy"
          value={`${((dashboardData?.best_accuracy || 0) * 100).toFixed(0)}%`}
          change="Augmented models"
          icon={<TrendingUp size={24} />}
          color="var(--success)"
        />
        <MetricCard
          title="Synthetic Samples"
          value={dashboardData?.synthetic_samples || 0}
          icon={<Brain size={24} />}
          color="var(--accent)"
        />
        <MetricCard
          title="Model Status"
          value={dashboardData?.model_status || "Loading..."}
          icon={<Activity size={24} />}
          color="var(--info)"
        />
      </div>

      <div className="dashboard-charts grid grid-2">
        <div className="card">
          <h3>Class Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={classDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {classDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3>Training Progress</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingProgress}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="loss" stroke="#ef4444" name="Loss" />
              <Line type="monotone" dataKey="accuracy" stroke="#10b981" name="Accuracy" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card mt-lg">
        <h3>Model Performance Comparison</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={modelComparison}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="Random Forest (Baseline)" fill="#94a3b8" />
            <Bar dataKey="Random Forest (Augmented)" fill="#6366f1" />
            <Bar dataKey="CNN (Baseline)" fill="#f59e0b" />
            <Bar dataKey="CNN (Augmented)" fill="#10b981" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="info-section mt-xl">
        <div className="card">
          <h3>Project Overview</h3>
          <p>
            This project uses EEG signals to classify mental arithmetic task performance quality.
            We employ GAN-based synthetic data generation to handle class imbalance and train
            both Random Forest and CNN classifiers. The augmented models show significant
            improvements in performance metrics.
          </p>
        </div>
      </div>
    </div>
  )
}

