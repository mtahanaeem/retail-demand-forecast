import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


class XGBoostModel:
    """XGBoost model for time series forecasting with feature engineering."""

    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators (int): Number of trees
            max_depth (int): Maximum depth of trees
            learning_rate (float): Learning rate
            random_state (int): Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.feature_names = None

    def prepare_features(self, df):
        """
        Prepare features for XGBoost model from DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with features
        
        Returns:
            pd.DataFrame: Features ready for training
            list: List of feature names
        """
        feature_cols = [
            'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
            'sales_roll_mean_7', 'sales_roll_mean_14', 'sales_roll_mean_28',
            'sales_roll_std_7', 'sales_roll_std_14', 'sales_roll_std_28',
            'day_of_week', 'month', 'is_weekend',
            'day_of_year', 'quarter'
        ]
        
        # Only keep columns that exist in the DataFrame
        existing_features = [col for col in feature_cols if col in df.columns]
        
        X = df[existing_features].copy()
        self.feature_names = existing_features
        
        return X

    def fit(self, X, y):
        """
        Fit XGBoost model to the data.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
        """
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            objective='reg:squarederror'
        )
        
        self.model.fit(X, y)
        return self

    def forecast(self, X):
        """
        Generate forecast from the fitted model.
        
        Args:
            X (pd.DataFrame): Features for prediction
        
        Returns:
            np.ndarray: Forecast values
        """
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        predictions = self.model.predict(X)
        return predictions

    def evaluate(self, actual, forecast):
        """
        Evaluate model performance.
        
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