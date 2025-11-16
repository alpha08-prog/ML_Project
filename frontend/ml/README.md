# EEG ML Project - Frontend

A modern, wholesome frontend for the EEG Mental Arithmetic Classification machine learning project.

## Features

- **Dashboard**: Overview of project metrics, class distribution, and training progress
- **Model Performance**: Detailed comparison between baseline and augmented models with comprehensive metrics
- **Data Visualization**: Interactive EEG signal visualization, UMAP embeddings, and channel analysis
- **Predictions**: Upload EEG files and get real-time predictions using trained models
- **About**: Comprehensive project documentation and methodology

## Tech Stack

- **React 19** with TypeScript
- **Vite** for fast development and building
- **React Router** for navigation
- **Recharts** for data visualization
- **Lucide React** for icons
- **Modern CSS** with CSS variables and animations

## Getting Started

### Prerequisites

- Node.js 18+ and npm

### Installation

```bash
cd frontend/ml
npm install
```

### Development

```bash
npm run dev
```

The application will be available at `http://localhost:5173`

### Build

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
src/
├── components/
│   ├── Layout.tsx          # Main layout with navigation
│   └── Layout.css
├── pages/
│   ├── Dashboard.tsx        # Overview dashboard
│   ├── ModelPerformance.tsx # Model comparison and metrics
│   ├── DataVisualization.tsx # EEG signals and UMAP plots
│   ├── Predictions.tsx      # Interactive prediction interface
│   └── About.tsx            # Project documentation
├── App.tsx                  # Main app with routing
├── App.css
├── index.css                # Global styles and design system
└── main.tsx                 # Entry point
```

## Design System

The frontend uses a comprehensive design system with:
- Modern color palette (primary, secondary, accent colors)
- Consistent spacing and typography
- Smooth animations and transitions
- Responsive grid layouts
- Accessible components

## Features Overview

### Dashboard
- Key metrics cards (Total Subjects, Best Accuracy, Synthetic Samples)
- Class distribution pie chart
- Training progress line chart
- Model performance comparison bar chart

### Model Performance
- Side-by-side metrics comparison
- Radar chart for multi-dimensional comparison
- Confusion matrix visualization
- Improvement tracking

### Data Visualization
- Interactive EEG signal viewer with channel selection
- Multi-channel overview
- UMAP embedding scatter plot
- Channel importance analysis
- KS test results visualization

### Predictions
- File upload interface
- Model selection (CNN or Random Forest)
- Real-time prediction results
- Recent predictions history
- Model information display

### About
- Problem statement
- Methodology timeline
- Dataset information
- Results summary
- Technologies used

## Customization

The design system can be customized by modifying CSS variables in `src/index.css`:

```css
:root {
  --primary: #6366f1;
  --secondary: #10b981;
  /* ... more variables */
}
```

## Notes

- The frontend currently uses mock data for demonstration
- To connect to a backend API, update the data fetching logic in each page component
- File upload in Predictions page is simulated - integrate with your ML inference API
