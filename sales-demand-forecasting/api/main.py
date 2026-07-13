from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_float(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return float(v)


def sanitize_metrics(d):
    return {k: safe_float(v) for k, v in d.items() if k in ('MAPE', 'RMSE', 'MAE')}

app = FastAPI(title="Retail Demand Forecasting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from baseline import BaselineModels
from sarima_model import SARIMAModel
from prophet_model import ProphetModel
from xgboost_model import XGBoostModel
from evaluate import evaluate_forecasts

class ForecastRequest(BaseModel):
    store_id: int
    product_family: str
    periods: int = 30
    use_backtest: bool = False

class MetricsRequest(BaseModel):
    store_id: int
    product_family: str
    n_test_periods: int = 12

# Load or create sample data
def load_or_create_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'store_sales.csv')
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Data loaded from {data_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")

    logger.info("Creating sample data...")
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
    stores = [1, 2, 3]
    product_families = ['A', 'B', 'C']
    data = []
    for store in stores:
        for product in product_families:
            for date in dates:
                season = np.sin(2 * np.pi * (date.dayofyear - 1) / 365) * 100
                noise = np.random.normal(0, 15)
                sales = max(0, int(season + noise + 50))
                data.append({
                    'store_id': store,
                    'product_family': product,
                    'date': date.strftime('%Y-%m-%d'),
                    'sales': sales
                })
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)
    return df

store_sales_df = load_or_create_data()

def get_series(store_id, product_family):
    df = store_sales_df[
        (store_sales_df['store_id'] == store_id) &
        (store_sales_df['product_family'] == product_family)
    ].copy()
    if df.empty:
        return None
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    return df['sales']

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/forecast")
async def get_forecast(request: ForecastRequest):
    try:
        series = get_series(request.store_id, request.product_family)
        if series is None:
            raise HTTPException(status_code=404, detail="No data found")

        historical = series.tolist()
        forecasts = {}

        # SARIMA
        try:
            m = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,7))
            m.fit(series)
            f = m.forecast(request.periods)
            forecasts['SARIMA'] = [safe_float(x) for x in f]
        except Exception as e:
            logger.warning(f"SARIMA failed: {e}")
            forecasts['SARIMA'] = [None] * request.periods

        # Prophet
        try:
            pdf = pd.DataFrame({'ds': series.index, 'y': series.values})
            m = ProphetModel()
            m.fit(pdf)
            f = m.forecast(request.periods)
            forecasts['Prophet'] = [safe_float(x) for x in f]
        except Exception as e:
            logger.warning(f"Prophet failed: {e}")
            forecasts['Prophet'] = [None] * request.periods

        # XGBoost
        try:
            from features import create_features
            df = series.to_frame(name='sales')
            df.index.name = 'date'
            df = df.reset_index()
            df = create_features(df)
            m = XGBoostModel()
            X = m.prepare_features(df)
            X = X.iloc[-request.periods:]
            y = df['sales'].iloc[-request.periods:]
            m.fit(X, y)
            f = m.forecast(X)
            forecasts['XGBoost'] = [safe_float(x) for x in f]
        except Exception as e:
            logger.warning(f"XGBoost failed: {e}")
            forecasts['XGBoost'] = [None] * request.periods

        # Baseline
        try:
            bm = BaselineModels()
            f = bm.forecast(series, request.periods)
            forecasts['Baseline'] = [safe_float(x) for x in f]
        except Exception as e:
            logger.warning(f"Baseline failed: {e}")

        return {
            "store_id": request.store_id,
            "product_family": request.product_family,
            "data": {
                "historical_actuals": historical,
                "forecasts": forecasts
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metrics")
async def get_metrics(request: MetricsRequest):
    try:
        series = get_series(request.store_id, request.product_family)
        if series is None:
            raise HTTPException(status_code=404, detail="No data found")

        n = request.n_test_periods
        actual = series[-n:].values
        history = series[:-n]

        if len(history) < 30:
            raise HTTPException(status_code=400, detail="Not enough data")

        results = {}

        # Baseline walk-forward
        try:
            bm = BaselineModels()
            bf = []
            h = history.tolist()
            for i in range(n):
                f = bm.forecast(pd.Series(h), 1)[0]
                bf.append(f)
                h.append(actual[i])
            eval_result = evaluate_forecasts(actual, Baseline=np.array(bf))
            results['Baseline'] = sanitize_metrics(eval_result['Baseline'])
        except Exception as e:
            logger.warning(f"Baseline metrics failed: {e}")

        # SARIMA
        try:
            m = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,7))
            m.fit(history)
            f = m.forecast(n)
            eval_result = evaluate_forecasts(actual, SARIMA=np.array(f))
            results['SARIMA'] = sanitize_metrics(eval_result['SARIMA'])
        except Exception as e:
            logger.warning(f"SARIMA metrics failed: {e}")

        return {
            "store_id": request.store_id,
            "product_family": request.product_family,
            "n_test_periods": n,
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stores")
async def get_stores():
    return {"stores": sorted(store_sales_df['store_id'].unique().tolist())}

@app.get("/product-families")
async def get_product_families():
    return {"product_families": sorted(store_sales_df['product_family'].unique().tolist())}

@app.get("/model-info")
async def get_model_info():
    return {
        "models": {
            "Baseline": {"description": "Naive forecast (last value repeated)", "parameters": []},
            "SARIMA": {"description": "Seasonal ARIMA with automatic tuning", "parameters": ["order", "seasonal_order"]},
            "Prophet": {"description": "Facebook Prophet with holiday effects", "parameters": ["seasonality_mode"]},
            "XGBoost": {"description": "XGBoost regressor with feature engineering", "parameters": ["n_estimators", "max_depth", "learning_rate"]}
        }
    }
