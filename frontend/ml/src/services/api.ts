/**
 * API Service for communicating with FastAPI backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface ApiResponse<T> {
  data?: T;
  error?: string;
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
  async getDashboard() {
    return this.request<any>('/api/dashboard');
  }

  // Model Performance
  async getPerformance() {
    return this.request<any>('/api/performance');
  }

  // Visualizations
  async getUMAP() {
    return this.request<any>('/api/visualization/umap');
  }

  async getChannelImportance() {
    return this.request<any>('/api/visualization/channel-importance');
  }

  async getEEGSignal(channel: number) {
    return this.request<any>(`/api/visualization/eeg-signal/${channel}`);
  }

  // Predictions
  async predict(file: File, modelType: 'rf' | 'cnn' = 'cnn') {
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

      return await response.json();
    } catch (error) {
      console.error('Prediction failed', error);
      throw error;
    }
  }

  // Dataset Info
  async getDatasetInfo() {
    return this.request<any>('/api/dataset/info');
  }

  // Health check
  async healthCheck() {
    return this.request<any>('/api/health');
  }
}

export const apiService = new ApiService();
export default apiService;

