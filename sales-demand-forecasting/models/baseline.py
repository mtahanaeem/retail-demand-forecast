import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


class BaselineModels:
    """Simple baseline forecasting models for comparison."""

    def naive_forecast(self, series, horizon=1):
        """
        Naive forecast that predicts the last value for future periods.
        
        Args:
            series (pd.Series): Time series data
            horizon (int): Number of periods to forecast
        
        Returns:
            np.ndarray: Forecast values
        """
        if len(series) == 0:
            return np.array([])
        
        last_value = series.iloc[-1]
        return np.full(horizon, last_value)

    def moving_average_forecast(self, series, horizon=1, window=7):
        """
        Moving average forecast using the specified window size.
        
        Args:
            series (pd.Series): Time series data
            horizon (int): Number of periods to forecast
            window (int): Window size for moving average
        
        Returns:
            np.ndarray: Forecast values
        """
        if len(series) < window:
            return self.naive_forecast(series, horizon)
        
        last_avg = series.iloc[-window:].mean()
        return np.full(horizon, last_avg)

    def evaluate_forecast(self, actual, forecast):
        """
        Evaluate forecast performance using common metrics.
        
        Args:
            actual (np.ndarray): Actual values
            forecast (np.ndarray): Forecast values
        
        Returns:
            dict: Dictionary containing MAE, MSE, and RMSE
        """
        mae = mean_absolute_error(actual, forecast)
        mse = mean_squared_error(actual, forecast)
        rmse = np.sqrt(mse)
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }

    def forecast(self, series, horizon=1):
        """
        Generate a forecast using naive method.
        
        Args:
            series (pd.Series): Time series data
            horizon (int): Number of periods to forecast
        
        Returns:
            np.ndarray: Forecast values
        """
        if len(series) == 0:
            return np.array([])
        last_value = series.iloc[-1] if hasattr(series, 'iloc') else series[-1]
        return np.full(horizon, last_value)

    def backtest(self, series, n_test, window=7):
        """
        Perform backtesting for baseline models.
        
        Args:
            series (pd.Series): Time series data
            n_test (int): Number of periods to backtest
            window (int): Window size for moving average
        
        Returns:
            dict: Dictionary containing actual and forecast values for all methods
        """
        if len(series) <= n_test:
            raise ValueError("Not enough data for backtesting")
        
        y_test = series[-n_test:].values
        history = series[:-n_test].tolist()
        
        naive_forecasts = []
        ma_forecasts = []
        
        for t in range(len(y_test)):
            naive_f = self.naive_forecast(pd.Series(history), horizon=1)[0]
            ma_f = self.moving_average_forecast(pd.Series(history), horizon=1, window=window)[0]
            
            naive_forecasts.append(naive_f)
            ma_forecasts.append(ma_f)
            
            # Update history with actual observation
            history.append(y_test[t])
        
        return {
            'actual': y_test,
            'naive_forecast': np.array(naive_forecasts),
            'ma_forecast': np.array(ma_forecasts)
        }