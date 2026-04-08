import { useState, useEffect, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from "recharts";
import { Eye, Maximize2, BarChart2 } from "lucide-react";
import apiService from "../services/api";

/* ── Types ────────────────────────────────────── */
interface EEGPoint {
  time: number;
  amplitude: number;
}
interface UMAPPoint {
  x: number;
  y: number;
  type: "real" | "synthetic" | "majority";
}
interface ChannelData {
  channels: string[];
  real: number[];
  synthetic: number[];
  importance: number[];
}

type View = "eeg" | "umap" | "distribution";

const CHANNEL_NAMES = [
  "Fp1",
  "Fp2",
  "F3",
  "F4",
  "C3",
  "C4",
  "P3",
  "P4",
  "O1",
  "O2",
  "F7",
  "F8",
  "T3",
  "T4",
  "T5",
  "T6",
  "Fz",
  "Cz",
  "Pz",
  "A1",
  "A2",
  "Fpz",
  "Oz",
];

/* ── Synthetic demo signal (fallback) ─────────── */
function generateDemoSignal(samples: number, seed: number): EEGPoint[] {
  const freq = 8 + (seed % 5);
  return Array.from({ length: samples }, (_, t) => ({
    time: t,
    amplitude:
      Math.sin((2 * Math.PI * freq * t) / samples) +
      Math.cos((Math.PI * freq * t) / samples) * 0.5 +
      (Math.random() - 0.5) * 0.2,
  }));
}

/* ── Component ────────────────────────────────── */
export default function DataVisualization() {
  const [view, setView] = useState<View>("eeg");
  const [selectedChannel, setSelectedChannel] = useState(0);
  const [eegData, setEegData] = useState<EEGPoint[]>([]);
  const [umapData, setUmapData] = useState<UMAPPoint[]>([]);
  const [channelData, setChannelData] = useState<ChannelData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /* EEG fetch */
  const fetchEEG = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.getEEGSignal(selectedChannel);
      setEegData(
        data.time.map((t: number, i: number) => ({ time: t, amplitude: data.amplitude[i] }))
      );
    } catch {
      setError("Failed to load EEG signal — showing demo data");
      setEegData(generateDemoSignal(128, selectedChannel));
    } finally {
      setLoading(false);
    }
  }, [selectedChannel]);

  /* UMAP fetch */
  const fetchUMAP = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.getUMAP();
      setUmapData(data.points ?? []);
    } catch {
      setError("Failed to load UMAP data");
    } finally {
      setLoading(false);
    }
  }, []);

  /* Channel importance fetch */
  const fetchChannelImportance = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.getChannelImportance();
      setChannelData(data);
    } catch {
      setError("Failed to load channel importance data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (view === "eeg") fetchEEG();
    if (view === "umap") fetchUMAP();
    if (view === "distribution") fetchChannelImportance();
  }, [view, fetchEEG, fetchUMAP, fetchChannelImportance]);

  /* Channel change re-fetches EEG */
  useEffect(() => {
    if (view === "eeg") fetchEEG();
  }, [selectedChannel]); // eslint-disable-line react-hooks/exhaustive-deps

  const views: { id: View; label: string; icon: React.ReactNode }[] = [
    { id: "eeg", label: "EEG Signals", icon: <Eye size={18} /> },
    { id: "umap", label: "UMAP Embedding", icon: <Maximize2 size={18} /> },
    { id: "distribution", label: "Channel Analysis", icon: <BarChart2 size={18} /> },
  ];

  return (
    <div className="data-visualization">
      <div className="page-header">
        <h1>Data Visualization</h1>
        <p className="subtitle">Explore EEG signals, UMAP embeddings, and feature distributions</p>
      </div>

      {/* View selector */}
      <div className="view-selector">
        {views.map(({ id, label, icon }) => (
          <button
            key={id}
            className={`view-btn ${view === id ? "active" : ""}`}
            onClick={() => setView(id)}
          >
            {icon}
            {label}
          </button>
        ))}
      </div>

      {/* ── EEG view ── */}
      {view === "eeg" && (
        <div className="visualization-section">
          <div className="card hover-float glass">
            <div className="chart-header">
              <h3>EEG Signal — {CHANNEL_NAMES[selectedChannel] ?? `Ch${selectedChannel}`}</h3>
              <div className="channel-selector">
                <label>Channel:</label>
                <select
                  className="channel-select"
                  value={selectedChannel}
                  onChange={(e) => setSelectedChannel(Number(e.target.value))}
                >
                  {CHANNEL_NAMES.map((name, idx) => (
                    <option key={idx} value={idx}>
                      {name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            {error && (
              <p
                style={{
                  color: "var(--warning)",
                  marginBottom: "var(--spacing-md)",
                  fontSize: "0.85rem",
                }}
              >
                {error}
              </p>
            )}
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={eegData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="time"
                  label={{ value: "Time (samples)", position: "insideBottom", offset: -5 }}
                />
                <YAxis label={{ value: "Amplitude (μV)", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="amplitude"
                  stroke="#6366f1"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
            <p className="chart-info">Baseline (60 s) + Task (60 s) · Downsampled to 128 Hz</p>
          </div>

          {/* Multi-channel demo */}
          <div className="card hover-float glass">
            <h3>Multi-Channel Overview (Demo)</h3>
            <ResponsiveContainer width="100%" height={420}>
              <LineChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                {CHANNEL_NAMES.slice(0, 8).map((name, idx) => (
                  <Line
                    key={idx}
                    type="monotone"
                    data={generateDemoSignal(128, idx).map((d) => ({
                      ...d,
                      [`ch${idx}`]: d.amplitude,
                    }))}
                    dataKey={`ch${idx}`}
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

      {/* ── UMAP view ── */}
      {view === "umap" && (
        <div className="visualization-section">
          <div className="card hover-float glass">
            <h3>UMAP Embedding</h3>
            <p className="chart-description">
              Dimensionality reduction showing real minority samples, GAN-generated synthetic
              samples, and majority class samples in 2D space.
            </p>
            {loading && (
              <div className="loading-container" style={{ minHeight: 200 }}>
                <p>Loading…</p>
              </div>
            )}
            {error && (
              <div style={{ color: "var(--error)", marginBottom: "var(--spacing-md)" }}>
                {error}
              </div>
            )}
            {!loading && !error && (
              <ResponsiveContainer width="100%" height={500}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="x"
                    label={{ value: "UMAP 1", position: "insideBottom", offset: -5 }}
                  />
                  <YAxis
                    dataKey="y"
                    label={{ value: "UMAP 2", angle: -90, position: "insideLeft" }}
                  />
                  <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                  <Legend />
                  <Scatter
                    name="Real Minority"
                    data={umapData.filter((d) => d.type === "real")}
                    fill="#10b981"
                  />
                  <Scatter
                    name="Synthetic"
                    data={umapData.filter((d) => d.type === "synthetic")}
                    fill="#6366f1"
                  />
                  <Scatter
                    name="Majority"
                    data={umapData.filter((d) => d.type === "majority")}
                    fill="#f59e0b"
                  />
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      )}

      {/* ── Channel analysis view ── */}
      {view === "distribution" && (
        <div className="visualization-section">
          <div className="card hover-float glass">
            <h3>Channel Importance Analysis</h3>
            <p className="chart-description">
              Feature importance per channel — real vs synthetic samples.
            </p>
            {loading && (
              <div className="loading-container" style={{ minHeight: 200 }}>
                <p>Loading…</p>
              </div>
            )}
            {error && (
              <div style={{ color: "var(--error)", marginBottom: "var(--spacing-md)" }}>
                {error}
              </div>
            )}
            {!loading && !error && channelData && (
              <ResponsiveContainer width="100%" height={400}>
                <LineChart
                  data={channelData.channels.map((ch, i) => ({
                    channel: ch,
                    real: channelData.real[i],
                    synthetic: channelData.synthetic[i],
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="channel" />
                  <YAxis label={{ value: "Mean |IG|", angle: -90, position: "insideLeft" }} />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="real"
                    stroke="#10b981"
                    name="Real"
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="synthetic"
                    stroke="#6366f1"
                    name="Synthetic"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>

          {channelData && (
            <div className="card hover-float glass">
              <h3>KS Test Results</h3>
              <p className="chart-description">
                Kolmogorov-Smirnov statistics per channel — real vs synthetic distribution
                similarity.
              </p>
              <div className="ks-results">
                {channelData.channels.slice(0, 10).map((ch, idx) => {
                  const val = channelData.importance[idx];
                  return (
                    <div key={idx} className="ks-item">
                      <span className="ks-channel">{ch}</span>
                      <div className="ks-bar-container">
                        <div
                          className="ks-bar"
                          style={{
                            width: `${Math.min(val * 300, 100)}%`,
                            background: val > 0.2 ? "#ef4444" : "#10b981",
                          }}
                        />
                      </div>
                      <span className="ks-value">{(val * 100).toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
