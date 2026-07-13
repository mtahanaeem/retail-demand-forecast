import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')


class SARIMAModel:
    """SARIMA model for time series forecasting."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), enforce_stationarity=False):
        """
        Initialize SARIMA model.
        
        Args:
            order (tuple): (p, d, q) for ARIMA part
            seasonal_order (tuple): (P, D, Q, S) for seasonal part
            enforce_stationarity (bool): Whether to enforce stationarity
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.model = None
        self.results = None

    def fit(self, series):
        """
        Fit SARIMA model to the time series.
        
        Args:
            series (pd.Series): Time series data
        """
        self.model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=False,
            start_params=[0, 0, 0, 0, 0, 0]
        )
        self.results = self.model.fit(disp=False)
        return self

    def forecast(self, horizon=1):
        """
        Generate forecast from the fitted model.
        
        Args:
            horizon (int): Number of periods to forecast
        
        Returns:
            np.ndarray: Forecast values
        """
        if self.results is None:
            raise ValueError("Model must be fitted before forecasting")
        return self.results.forecast(horizon)

    def evaluate(self, actual, forecast):
        """
        Evaluate model performance.
        
        Args:
            actual (np.ndarray): Actual values
            forecast (np.ndarray): Forecast values
        
        Returns:
            dict: Dictionary containing MAE, MSE, and RMSE
        """
        mae = np.mean(np.abs(actual - forecast))
        mse = np.mean((actual - forecast) ** 2)
        rmse = np.sqrt(mse)
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }