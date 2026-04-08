import { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { CheckCircle } from "lucide-react";
import apiService from "../services/api";

/* ── Types ────────────────────────────────────── */
interface ModelMetrics {
  accuracy: number;
  roc_auc: number;
  pr_auc: number;
  precision: number;
  recall: number;
  f1: number;
}
interface PerformanceData {
  random_forest: { baseline: ModelMetrics; augmented: ModelMetrics };
  cnn?: { baseline: ModelMetrics; augmented: ModelMetrics };
}

type FlatMetrics = {
  accuracy: number;
  rocAuc: number;
  prAuc: number;
  precision: number;
  recall: number;
  f1: number;
};

function toFlat(m: ModelMetrics): FlatMetrics {
  return {
    accuracy: m.accuracy,
    rocAuc: m.roc_auc,
    prAuc: m.pr_auc,
    precision: m.precision,
    recall: m.recall,
    f1: m.f1,
  };
}

/* ── Component ────────────────────────────────── */
export default function ModelPerformance() {
  const [selectedModel, setSelectedModel] = useState<"rf" | "cnn">("rf");
  const [data, setData] = useState<PerformanceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiService
      .getPerformance()
      .then(setData)
      .catch((err) => {
        console.error(err);
        setError("Failed to load performance data");
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading)
    return (
      <div className="loading-container">
        <p>Loading performance data…</p>
      </div>
    );
  if (error || !data)
    return (
      <div className="loading-container">
        <p style={{ color: "var(--error)" }}>{error ?? "No data"}</p>
      </div>
    );

  const hasCNN = !!data.cnn;

  const rfFlat = {
    baseline: toFlat(data.random_forest.baseline),
    augmented: toFlat(data.random_forest.augmented),
  };
  const cnnFlat = hasCNN
    ? { baseline: toFlat(data.cnn!.baseline), augmented: toFlat(data.cnn!.augmented) }
    : null;

  const { baseline, augmented } = selectedModel === "cnn" && cnnFlat ? cnnFlat : rfFlat;

  const METRIC_LABELS: [keyof FlatMetrics, string][] = [
    ["accuracy", "Accuracy"],
    ["rocAuc", "ROC-AUC"],
    ["prAuc", "PR-AUC"],
    ["precision", "Precision"],
    ["recall", "Recall"],
    ["f1", "F1-Score"],
  ];

  const comparisonData = METRIC_LABELS.map(([key, label]) => ({
    metric: label,
    baseline: baseline[key],
    augmented: augmented[key],
  }));

  const improvements = METRIC_LABELS.map(([key, label]) => ({
    label,
    delta: ((augmented[key] - baseline[key]) * 100).toFixed(1),
  }));

  return (
    <div className="model-performance">
      <div className="page-header">
        <h1>Model Performance</h1>
        <p className="subtitle">Compare baseline vs augmented model metrics</p>
      </div>

      {/* Model toggle */}
      <div className="model-selector">
        <button
          className={selectedModel === "rf" ? "active" : ""}
          onClick={() => setSelectedModel("rf")}
        >
          Random Forest
        </button>
        <button
          className={selectedModel === "cnn" ? "active" : ""}
          onClick={() => hasCNN && setSelectedModel("cnn")}
          disabled={!hasCNN}
        >
          CNN {!hasCNN && "(Not Available)"}
        </button>
      </div>

      {/* Bar chart */}
      <div className="card">
        <h3>Baseline vs Augmented</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={comparisonData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="metric" />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
            <Legend />
            <Bar dataKey="baseline" fill="#94a3b8" name="Baseline" />
            <Bar dataKey="augmented" fill="#10b981" name="Augmented" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Radar chart */}
      <div className="card mt-lg">
        <h3>Radar Comparison</h3>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={comparisonData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="metric" />
            <PolarRadiusAxis domain={[0, 1]} />
            <Radar
              name="Baseline"
              dataKey="baseline"
              stroke="#94a3b8"
              fill="#94a3b8"
              fillOpacity={0.4}
            />
            <Radar
              name="Augmented"
              dataKey="augmented"
              stroke="#10b981"
              fill="#10b981"
              fillOpacity={0.4}
            />
            <Legend />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Improvements */}
      <div className="improvements-grid mt-lg">
        {improvements.map(({ label, delta }) => (
          <div key={label} className="improvement-item">
            <CheckCircle size={16} />
            <span>
              {label}: <strong>+{delta}%</strong>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
