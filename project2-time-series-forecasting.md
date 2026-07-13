# Build Spec: Retail Demand Forecasting Pipeline

> Give this file to an AI coding assistant (Claude Code, Cursor, etc.) as the project brief. It describes exactly what to build, in what order, and what "done" looks like.

## Context
Build a complete time series forecasting project for a Data Science job application. The project must show statistical decomposition, multiple modeling approaches, and — most importantly — time-aware evaluation (walk-forward validation, not random train/test splits).

## Dataset
Use the **Store Sales - Time Series Forecasting** dataset (public, available on Kaggle: `store-sales-time-series-forecasting`). It contains daily sales across multiple stores and product families, with holiday and promotion data.

## Tech Stack
Python 3.11, pandas, statsmodels, prophet, XGBoost, FastAPI (backend), React + Vite + Tailwind + Recharts (frontend)

## Required Project Structure
```
sales-demand-forecasting/
├── data/                    # raw + processed data (gitignored except sample)
├── notebooks/
│   └── 01_decomposition.ipynb
├── src/
│   ├── preprocessing.py     # resampling, merging calendar/holiday data
│   ├── features.py          # lag features, rolling means, calendar features
│   ├── models/
│   │   ├── baseline.py      # naive + moving average
│   │   ├── sarima_model.py
│   │   ├── prophet_model.py
│   │   └── xgboost_model.py
│   ├── backtest.py          # walk-forward validation logic
│   └── evaluate.py          # MAPE, RMSE, MAE comparison across models
├── api/
│   └── main.py               # FastAPI app exposing /forecast and /metrics endpoints (CORS enabled)
├── frontend/                 # React app (Vite)
│   ├── src/
│   │   ├── components/
│   │   │   ├── StoreSelector.jsx     # dropdown for store/product family
│   │   │   ├── ForecastChart.jsx     # actual vs forecast line chart (Recharts)
│   │   │   └── MetricsTable.jsx      # MAPE/RMSE/MAE comparison table
│   │   ├── api.js            # fetch wrapper calling FastAPI endpoints
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── models/                  # saved model artifacts
├── tests/
│   └── test_backtest.py
├── requirements.txt
└── README.md
```

## Build Order (do these in sequence)

1. **Data loading + decomposition** (`notebooks/01_decomposition.ipynb`)
   - Load and resample to daily/weekly frequency per store/product family
   - Run `seasonal_decompose` to show trend/seasonality/residual
   - Run Augmented Dickey-Fuller test for stationarity
   - Visualize seasonality by day-of-week, month, and around holidays

2. **Feature engineering** (`src/preprocessing.py`, `src/features.py`)
   - Merge holiday/promotion calendar data
   - Create lag features (t-1, t-7, t-14, t-28)
   - Create rolling mean/std features (7-day, 28-day windows)
   - Add calendar features: day of week, month, is_holiday, is_weekend

3. **Modeling — build up in complexity**
   - `models/baseline.py`: naive forecast (last value) and moving average
   - `models/sarima_model.py`: SARIMA with seasonal order tuned via grid search or `auto_arima`
   - `models/prophet_model.py`: Prophet with holiday effects included
   - `models/xgboost_model.py`: XGBoost regressor using lag + rolling + calendar features

4. **Walk-forward backtesting** (`src/backtest.py`)
   - Implement expanding-window walk-forward validation (train on past N periods, predict next period, roll forward)
   - Explicitly do NOT use random train/test splits — this must be time-ordered
   - Run all four models through the same backtest windows for a fair comparison

5. **Evaluation** (`src/evaluate.py`)
   - Compute MAPE, RMSE, MAE per model across all backtest folds
   - Output a comparison table (model × metric)
   - Plot forecast vs. actual for each model, highlighting failure points (holidays, demand spikes)

6. **API** (`api/main.py`)
   - FastAPI endpoints: `/forecast?store_id=&product_family=` returns historical actuals + forecasts from all models; `/metrics` returns the comparison table
   - Enable CORS for the React frontend

7. **Frontend** (`frontend/`)
   - Scaffold with Vite (`npm create vite@latest frontend -- --template react`) + Tailwind
   - `StoreSelector.jsx`: dropdowns to pick store and product family, triggers a re-fetch
   - `api.js`: calls `/forecast` and `/metrics` endpoints
   - `ForecastChart.jsx`: line chart (Recharts) overlaying historical actuals with each model's forecast, highlighting holidays/spikes
   - `MetricsTable.jsx`: renders the MAPE/RMSE/MAE comparison table below the chart
   - `App.jsx`: wires selector → API calls → chart + table, with loading states

8. **Packaging**
   - `requirements.txt` with pinned versions
   - Basic unit test for the walk-forward split logic (verify no data leakage)

9. **README.md** must include:
   - Problem statement (1 paragraph)
   - Why walk-forward validation was used instead of random splits (this is the key differentiator — spell it out clearly)
   - Model comparison table with metrics
   - Final recommendation: which model to use in production and why (accuracy vs. interpretability vs. compute cost)
   - How to run locally (`pip install -r requirements.txt`, `uvicorn api.main:app --reload`, then `cd frontend && npm install && npm run dev`)

## Definition of Done
- All four models run through identical walk-forward backtest folds
- Comparison table with MAPE/RMSE/MAE for each model
- React frontend displays forecast vs. actual interactively, with store/product selection working end-to-end
- README clearly explains the walk-forward validation choice and the final model recommendation
