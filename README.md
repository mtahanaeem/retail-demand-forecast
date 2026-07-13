# 📈 Retail Demand Forecasting

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?logo=react&logoColor=61DAFB)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?logo=xgboost&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

A **retail demand forecasting dashboard** that compares 4 forecasting models side-by-side — Baseline, SARIMA, Prophet, and XGBoost — with a modern React frontend and FastAPI backend.

---

## ✨ Features

- 📊 **Interactive Dashboard** — Select stores & product families, view forecasts instantly
- 🤖 **4 Forecasting Models** — Compare Baseline, SARIMA, Prophet, and XGBoost
- 🔄 **Walk-Forward Validation** — Industry-standard time series evaluation (no data leakage)
- 🌙 **Dark Theme UI** — Sleek, modern interface powered by Recharts
- ⚡ **Real-Time API** — FastAPI backend with CORS support
- 📉 **Performance Metrics** — MAPE, RMSE, and MAE for every model

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python) |
| Frontend | React + Vite |
| Charts | Recharts |
| Models | SARIMA, Prophet, XGBoost, scikit-learn |
| Validation | Walk-forward (expanding window) |

---

## 📁 Project Structure

```
retail-demand-forecast/
├── api/
│   └── main.py                 # FastAPI endpoints
├── models/
│   ├── baseline.py             # Naive / moving average
│   ├── sarima_model.py         # Seasonal ARIMA
│   ├── prophet_model.py        # Facebook Prophet
│   └── xgboost_model.py        # XGBoost with feature engineering
├── src/
│   ├── preprocessing.py        # Data loading & cleaning
│   ├── features.py             # Feature engineering (lags, rolling stats)
│   ├── backtest.py             # Walk-forward validation
│   └── evaluate.py             # MAPE, RMSE, MAE
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ForecastChart.jsx
│   │   │   ├── MetricsComparison.jsx
│   │   │   └── SummaryCards.jsx
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── tests/
│   └── test_backtest.py
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

### Prerequisites

- Python 3.11+
- Node.js 18+

### Installation

```bash
git clone https://github.com/mtahanaeem/retail-demand-forecast.git
cd retail-demand-forecast

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Run Backend

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Run Frontend (separate terminal)

```bash
cd frontend
npm run dev
```

### Access

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Health Check | http://localhost:8000/health |

---

## 🧠 How It Works

1. **Select** a store and product family from the dropdown
2. **View** historical sales plotted alongside forecasts from all 4 models
3. **Compare** model performance metrics (MAPE, RMSE, MAE)
4. **Identify** the best model highlighted in the dashboard

All models are evaluated using **walk-forward validation** — no random train/test splits, ensuring realistic performance estimates.

---

## 👤 Author

**Muhammad Taha Naeem**

- 📧 muhamadtahanaeem.pro@gmail.com
- 🐙 [mtahanaeem](https://github.com/mtahanaeem)
