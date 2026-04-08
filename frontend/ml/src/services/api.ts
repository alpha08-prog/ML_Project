/**
 * API Service for communicating with FastAPI backend
 */

const API_BASE_URL = '';

interface ClassDistribution {
  good: number;
  bad: number;
}

interface DashboardResponse {
  total_subjects: number;
  best_accuracy: number;
  synthetic_samples: number;
  class_distribution: ClassDistribution;
  model_status: string;
}

interface ModelMetrics {
  accuracy: number;
  roc_auc: number;
  pr_auc: number;
  precision: number;
  recall: number;
  f1: number;
}

interface PerformanceResponse {
  random_forest: {
    baseline: ModelMetrics;
    augmented: ModelMetrics;
  };
  cnn?: {
    baseline: ModelMetrics;
    augmented: ModelMetrics;
  };
}

interface UMAPPoint {
  x: number;
  y: number;
  type: 'real' | 'synthetic' | 'majority';
}

interface UMAPResponse {
  points: UMAPPoint[];
}

interface ChannelImportanceResponse {
  channels: string[];
  importance: number[];
  real: number[];
  synthetic: number[];
}

interface EEGSignalResponse {
  time: number[];
  amplitude: number[];
}

interface PredictionResponse {
  class: string;
  probability: number;
  confidence: string;
}

interface DatasetInfo {
  total_subjects: number;
  synthetic_samples: number;
  class_distribution: ClassDistribution;
  channels: number;
  sequence_length: number;
}

interface HealthResponse {
  status: string;
  models_loaded: boolean;
  torch_available: boolean;
  cnn_available: boolean;
  rf_available: boolean;
}

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // Dashboard
  async getDashboard(): Promise<DashboardResponse> {
    return this.request<DashboardResponse>('/api/dashboard');
  }

  // Model Performance
  async getPerformance(): Promise<PerformanceResponse> {
    return this.request<PerformanceResponse>('/api/performance');
  }

  // Visualizations
  async getUMAP(): Promise<UMAPResponse> {
    return this.request<UMAPResponse>('/api/visualization/umap');
  }

  async getChannelImportance(): Promise<ChannelImportanceResponse> {
    return this.request<ChannelImportanceResponse>('/api/visualization/channel-importance');
  }

  async getEEGSignal(channel: number): Promise<EEGSignalResponse> {
    return this.request<EEGSignalResponse>(`/api/visualization/eeg-signal/${channel}`);
  }

  // Predictions
  async predict(file: File, modelType: 'rf' | 'cnn' = 'cnn'): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', modelType);

    try {
      const response = await fetch(`${this.baseUrl}/api/predict?model_type=${modelType}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json() as PredictionResponse;
    } catch (error) {
      console.error('Prediction failed', error);
      throw error;
    }
  }

  // Dataset Info
  async getDatasetInfo(): Promise<DatasetInfo> {
    return this.request<DatasetInfo>('/api/dataset/info');
  }

  // Health check
  async healthCheck(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/api/health');
  }
}

export const apiService = new ApiService();
export default apiService;
