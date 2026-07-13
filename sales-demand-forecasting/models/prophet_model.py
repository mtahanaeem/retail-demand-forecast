import pandas as pd
import numpy as np
from prophet import Prophet


class ProphetModel:
    """Prophet model for time series forecasting with holiday effects."""

    def __init__(self, seasonality_mode='additive', daily_seasonality=True, yearly_seasonality=True):
        """
        Initialize Prophet model.
        
        Args:
            seasonality_mode (str): 'additive' or 'multiplicative'
            daily_seasonality (bool): Whether to include daily seasonality
            yearly_seasonality (bool): Whether to include yearly seasonality
        """
        self.seasonality_mode = seasonality_mode
        self.daily_seasonality = daily_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.model = None

    def prepare_data(self, df):
        """
        Prepare data for Prophet model.
        
        Args:
            df (pd.DataFrame): DataFrame with 'date' and 'sales' columns
        
        Returns:
            pd.DataFrame: DataFrame prepared for Prophet
        """
        df_prophet = df.copy()
        df_prophet = df_prophet.rename(columns={'date': 'ds', 'sales': 'y'})
        
        # Add holiday and promotion features
        if 'is_holiday' in df_prophet.columns:
            df_prophet['holiday'] = df_prophet['is_holiday'].astype(int)
        if 'promotion' in df_prophet.columns:
            df_prophet['promotion'] = df_prophet['promotion']
        
        return df_prophet

    def fit(self, df):
        """
        Fit Prophet model to the data.
        
        Args:
            df (pd.DataFrame): DataFrame with 'ds' and 'y' columns
        """
        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=self.daily_seasonality,
            yearly_seasonality=self.yearly_seasonality
        )
        
        # Add country holidays if available
        if 'is_holiday' in df.columns:
            holidays = df[df['is_holiday'] == True].copy()
            holidays = holidays.rename(columns={'ds': 'holiday', 'sales': 'edge'} if 'sales' in holidays.columns else {'ds': 'holiday'})
            if len(holidays) > 0:
                self.model.add_events(holidays, name='holiday')
        
        self.model.fit(df)
        return self

    def forecast(self, horizon=1):
        """
        Generate forecast from the fitted model.
        
        Args:
            horizon (int): Number of periods to forecast
        
        Returns:
            np.ndarray: Forecast values
        """
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        future = self.model.make_future_dataframe(periods=horizon)
        forecast = self.model.predict(future)
        
        return forecast['yhat'].values[-horizon:]

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