import { useState } from 'react'
import { Upload, FileText, CheckCircle, XCircle, Loader } from 'lucide-react'
import apiService from '../services/api'
import './Predictions.css'

export default function Predictions() {
  const [selectedModel, setSelectedModel] = useState<'rf' | 'cnn'>('cnn')
  const [file, setFile] = useState<File | null>(null)
  const [predicting, setPredicting] = useState(false)
  const [prediction, setPrediction] = useState<{
    class: string
    probability: number
    confidence: string
  } | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setPrediction(null)
    }
  }

  const handlePredict = async () => {
    if (!file) return

    setPredicting(true)
    try {
      const result = await apiService.predict(file, selectedModel)
      setPrediction(result)
    } catch (error) {
      console.error('Prediction error:', error)
      alert('Failed to make prediction. Make sure the backend is running and the file is valid.')
    } finally {
      setPredicting(false)
    }
  }

  const samplePredictions = [
    { subject: 'Subject 01', model: 'CNN Augmented', prediction: 'Good Quality', probability: 0.87, actual: 'Good Quality', correct: true },
    { subject: 'Subject 04', model: 'CNN Augmented', prediction: 'Bad Quality', probability: 0.72, actual: 'Bad Quality', correct: true },
    { subject: 'Subject 07', model: 'Random Forest Augmented', prediction: 'Good Quality', probability: 0.81, actual: 'Good Quality', correct: true },
    { subject: 'Subject 10', model: 'CNN Augmented', prediction: 'Bad Quality', probability: 0.68, actual: 'Bad Quality', correct: true },
  ]

  return (
    <div className="predictions">
      <div className="page-header">
        <h1>Predictions</h1>
        <p className="subtitle">Upload EEG data and get real-time predictions</p>
      </div>

      <div className="predictions-layout grid grid-2">
        <div className="card hover-float glass">
          <h3>Make a Prediction</h3>
          <div className="prediction-form">
            <div className="model-selector-group">
              <label>Select Model:</label>
              <div className="model-radio-group">
                <label className="radio-label">
                  <input
                    type="radio"
                    value="cnn"
                    checked={selectedModel === 'cnn'}
                    onChange={(e) => setSelectedModel(e.target.value as 'rf' | 'cnn')}
                  />
                  <span>CNN Augmented</span>
                </label>
                <label className="radio-label">
                  <input
                    type="radio"
                    value="rf"
                    checked={selectedModel === 'rf'}
                    onChange={(e) => setSelectedModel(e.target.value as 'rf' | 'cnn')}
                  />
                  <span>Random Forest Augmented</span>
                </label>
              </div>
            </div>

            <div className="file-upload-area">
              <input
                type="file"
                id="file-upload"
                accept=".edf,.csv"
                onChange={handleFileChange}
                className="file-input"
              />
              <label htmlFor="file-upload" className="file-upload-label">
                {file ? (
                  <div className="file-selected">
                    <FileText size={24} />
                    <span>{file.name}</span>
                  </div>
                ) : (
                  <div className="file-upload-placeholder">
                    <Upload size={32} />
                    <span>Click to upload EEG file (.edf or .csv)</span>
                    <small>Supports EDF format or preprocessed CSV</small>
                  </div>
                )}
              </label>
            </div>

            <button
              className="btn btn-primary predict-btn hover-glow"
              onClick={handlePredict}
              disabled={!file || predicting}
            >
              {predicting ? (
                <>
                  <Loader className="spinner" size={18} />
                  Predicting...
                </>
              ) : (
                'Predict'
              )}
            </button>
          </div>

          {prediction && (
            <div className="prediction-result">
              <h4>Prediction Result</h4>
              <div className={`result-card ${prediction.class.includes('Good') ? 'result-good' : 'result-bad'}`}>
                <div className="result-header">
                  {prediction.class.includes('Good') ? (
                    <CheckCircle size={32} className="result-icon" />
                  ) : (
                    <XCircle size={32} className="result-icon" />
                  )}
                  <div>
                    <div className="result-class">{prediction.class}</div>
                    <div className="result-confidence">Confidence: {prediction.confidence}</div>
                  </div>
                </div>
                <div className="result-probability">
                  <div className="probability-bar">
                    <div
                      className="probability-fill"
                      style={{ width: `${prediction.probability * 100}%` }}
                    />
                  </div>
                  <span className="probability-value">{(prediction.probability * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="card hover-float glass">
          <h3>Recent Predictions</h3>
          <div className="predictions-list">
            {samplePredictions.map((pred, idx) => (
              <div key={idx} className="prediction-item">
                <div className="prediction-item-header">
                  <span className="prediction-subject">{pred.subject}</span>
                  {pred.correct ? (
                    <CheckCircle size={18} className="prediction-correct" />
                  ) : (
                    <XCircle size={18} className="prediction-incorrect" />
                  )}
                </div>
                <div className="prediction-item-details">
                  <div className="prediction-model">{pred.model}</div>
                  <div className="prediction-info">
                    <span className={`prediction-label ${pred.prediction.includes('Good') ? 'label-good' : 'label-bad'}`}>
                      {pred.prediction}
                    </span>
                    <span className="prediction-prob">{(pred.probability * 100).toFixed(0)}%</span>
                  </div>
                  <div className="prediction-actual">Actual: {pred.actual}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card mt-lg">
        <h3>Model Information</h3>
        <div className="model-info-grid grid grid-2">
          <div className="model-info-item">
            <h4>CNN Augmented</h4>
            <ul>
              <li>Accuracy: 88%</li>
              <li>ROC-AUC: 0.86</li>
              <li>F1-Score: 0.85</li>
              <li>Input: (Channels, Time) - (23, 128)</li>
            </ul>
          </div>
          <div className="model-info-item">
            <h4>Random Forest Augmented</h4>
            <ul>
              <li>Accuracy: 85%</li>
              <li>ROC-AUC: 0.83</li>
              <li>F1-Score: 0.81</li>
              <li>Input: Flattened features (2944)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

