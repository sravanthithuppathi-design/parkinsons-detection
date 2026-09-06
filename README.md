# 🧠 PD Detect — Parkinson's Disease Voice Analysis Platform

> **Research & Educational Use Only.** This tool does not provide medical diagnoses.  
> Always consult a qualified neurologist for clinical evaluation.

A production-quality, AI-powered web platform for Parkinson's Disease screening through voice biomarker analysis. Built with a Python ML backend and React frontend, deployed globally.

---

## 🌐 Live Application

| Service | URL |
|---|---|
| **Web App** | **https://pd-detect.vercel.app** |
| **API** | https://pd-detect-api.onrender.com |
| **API Docs** | https://pd-detect-api.onrender.com/docs |
| **Source Code** | https://github.com/sujan7989/pd-detect |

> The client only needs to open **https://pd-detect.vercel.app** — no installation required.

---

## ✨ Features

### Audio Processing Pipeline
| Stage | Implementation |
|---|---|
| **Format Support** | WAV, MP3, OGG, WebM, FLAC via ffmpeg + soundfile |
| **Noise Reduction** | Multi-stage: spectral subtraction, temporal noise gate, harmonic enhancement |
| **Preprocessing** | Band-pass filter (80–8000 Hz), silence trimming, automatic gain normalization |
| **Quality Assessment** | SNR estimation, pitch quality, jitter/shimmer quality scoring |
| **Quality Score** | 0–100% rating returned with every analysis |
| **Corruption Detection** | Rejects physiologically impossible features caused by extreme noise |
| **Recording Validation** | `/validate-audio` endpoint for pre-analysis quality check |

### ML Pipeline
| Component | Details |
|---|---|
| **Dataset** | UCI Parkinson's Dataset (195 samples, 22 clinical features) |
| **Features Extracted** | 68+ acoustic biomarkers per recording |
| **Models** | Random Forest + XGBoost + SVM + Gradient Boosting (4-model ensemble) |
| **Ensemble** | Soft-voting VotingClassifier |
| **Accuracy** | 92.3% cross-validation accuracy |
| **Risk Levels** | Very Low / Low / Moderate / High / Very High (5-level) |
| **Quality Override** | Noise-corrupted recordings are flagged and prediction is adjusted |

### Speech Biomarkers Extracted
- **MFCC**: 13 coefficients + deltas + delta-deltas (39 features)
- **Jitter**: Local, Absolute, RAP, PPQ5, DDP (5 variants)
- **Shimmer**: Local, dB, APQ3, APQ5, APQ11, DDA (6 variants)
- **Pitch**: Mean, Std, stability (via autocorrelation + zero-crossing)
- **HNR**: Harmonic-to-Noise Ratio (cepstrum method)
- **ZCR**: Zero Crossing Rate
- **Spectral**: Centroid, Rolloff, Bandwidth
- **Energy**: RMS (frame-level + global)
- **Nonlinear**: RPDE proxy, DFA proxy, Spread1, Spread2, PPE
- **Quality**: SNR estimate, recording duration

### User Interface
- 🌙 **Dark/Light mode toggle** in navbar
- 🎙️ **Live waveform** visualization during recording
- 📊 **Interactive charts**: Radar chart (patient vs healthy reference), Feature bar chart, Feature importance
- 🎯 **Confidence ring** with animated SVG gauge
- 📋 **Quality score badge** on every result
- ⚠️ **Noise warning** shown when recording quality is poor
- 📱 **Mobile responsive** — works on phone and tablet
- ✨ **Particle background animation** with ambient color orbs
- 📈 **Analysis history** saved in browser localStorage (persists across sessions)
- 📊 **Dashboard** with charts — pie distribution, confidence trend, feature averages

---

## 🚀 Quick Start (Client-Friendly)

### Option 1 — Live URL (No Setup)
```
Open: https://pd-detect.vercel.app
```

### Option 2 — From ZIP (Local)
1. Extract `PD_Detect_Client.zip`
2. Install prerequisites:
   - Python 3.10+ → https://python.org/downloads (check "Add to PATH")
   - Node.js 18+ → https://nodejs.org
3. Double-click **`START_BACKEND.bat`** — wait for `✅ Models loaded`
4. Double-click **`START_FRONTEND.bat`** — wait for `➜ Local: http://localhost:5173`
5. Open browser → `http://localhost:5173`

---

## 🛠️ Manual Setup

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
# Models are pre-trained — no retraining needed
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### Retrain Models (Optional)
```bash
cd backend
venv\Scripts\python model_trainer.py
# Downloads UCI dataset, trains RF+XGBoost+SVM+GB, saves to models/
```

---

## 📁 Project Structure

```
parkinsons-detection/
├── backend/
│   ├── main.py                     # FastAPI — all API endpoints
│   ├── feature_extractor.py        # Audio → 68+ features (pure numpy/scipy)
│   ├── model_trainer.py            # Train ensemble on UCI dataset
│   ├── requirements.txt
│   ├── Dockerfile                  # For containerized deployment
│   ├── render.yaml                 # Render deployment config
│   └── models/
│       ├── ensemble.pkl            ✅ Pre-trained
│       ├── random_forest.pkl       ✅ Pre-trained
│       ├── xgboost.pkl             ✅ Pre-trained
│       ├── svm.pkl                 ✅ Pre-trained
│       ├── gradient_boost.pkl      ✅ Pre-trained
│       ├── scaler.pkl              ✅ Pre-trained
│       ├── feature_importance.json
│       └── uci_feature_stats.json
└── frontend/
    ├── src/
    │   ├── App.jsx                 # Main app + dark/light mode
    │   ├── api/client.js           # API + localStorage history
    │   ├── pages/
    │   │   ├── AnalyzePage.jsx     # Recording + analysis
    │   │   ├── DashboardPage.jsx   # Analytics charts
    │   │   ├── HistoryPage.jsx     # Past analyses
    │   │   └── AboutPage.jsx       # Documentation
    │   └── components/
    │       ├── AudioRecorder.jsx   # Record/upload + quality tip
    │       ├── WaveformVisualizer.jsx
    │       ├── ResultPanel.jsx     # Results + quality score + warnings
    │       ├── ConfidenceRing.jsx  # Animated SVG gauge
    │       ├── AcousticCharts.jsx  # Radar + bar + importance charts
    │       └── ParticleField.jsx   # Canvas background animation
    ├── vercel.json
    └── package.json
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server health + model status |
| POST | `/analyze` | Full voice analysis → prediction |
| POST | `/validate-audio` | Quick quality check only (no ML) |
| POST | `/report` | Generate PDF report |
| GET | `/history` | Session history |
| DELETE | `/history` | Clear all history |
| GET | `/stats` | Aggregate statistics |

### POST /analyze — Response Fields
```json
{
  "id": "A1B2C3D4",
  "timestamp": "2025-01-15 10:30:00 UTC",
  "prediction": "Healthy",
  "confidence": 72.5,
  "probability": 0.275,
  "risk_percent": 27.5,
  "risk_level": "Low",
  "quality_score": 83.0,
  "is_corrupted": false,
  "snr_db": 18.5,
  "features": { "pitch_mean": 182.3, "jitter_local": 0.0038, ... },
  "feature_importance": { "mfcc_5_mean": { "importance": 0.093 }, ... },
  "recommendations": [ "Voice analysis appears within typical parameters." ],
  "model_used": "Ensemble (RF + XGBoost + SVM + GB)"
}
```

### POST /validate-audio — Response
```json
{
  "quality_score": 83.0,
  "is_corrupted": false,
  "snr_db": 18.5,
  "duration_sec": 5.2,
  "issues": [],
  "recommendation": "Good quality — proceed with analysis"
}
```

---

## 🤖 ML Models

| Model | CV Accuracy |
|---|---|
| Random Forest (300 trees) | ~94.9% |
| XGBoost (300 estimators) | ~94.9% |
| SVM (Calibrated, RBF) | ~87.2% |
| Gradient Boosting | ~91.3% |
| **Ensemble (soft voting)** | **~92.3%** |

Trained on UCI Parkinson's Dataset:  
> Little MA et al. (2007). *Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection.* BioMedical Engineering OnLine.

---

## 🛡️ Disclaimer

This software is intended **solely for research and educational purposes**. It:
- ❌ Does **not** provide medical diagnoses
- ❌ Is **not** FDA approved or clinically validated
- ❌ Should **not** replace professional medical advice
- ✅ Uses real UCI clinical data for training
- ✅ Provides calibrated probability estimates
- ✅ Always displays research/educational disclaimer

**Always consult a board-certified neurologist for Parkinson's Disease evaluation.**

---

## 📜 License
MIT License — Free for research and educational use.
