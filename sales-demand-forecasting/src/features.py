import pandas as pd
import numpy as np


def add_lag_features(df, lags=[1, 7, 14, 28]):
    """
    Add lag features to the dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame with sales data
        lags (list): List of lag periods to create
    
    Returns:
        pd.DataFrame: DataFrame with lag features added
    """
    df = df.sort_values('date')
    
    for lag in lags:
        df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
    
    return df


def add_rolling_features(df, windows=[7, 14, 28]):
    """
    Add rolling mean and standard deviation features.
    
    Args:
        df (pd.DataFrame): DataFrame with sales data
        windows (list): List of window sizes for rolling statistics
    
    Returns:
        pd.DataFrame: DataFrame with rolling features added
    """
    df = df.sort_values('date')
    
    for window in windows:
        df[f'sales_roll_mean_{window}'] = df['sales'].rolling(window=window).mean()
        df[f'sales_roll_std_{window}'] = df['sales'].rolling(window=window).std()
    
    return df


def add_calendar_features(df):
    """
    Add calendar-based features to the dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame with date column
    
    Returns:
        pd.DataFrame: DataFrame with calendar features added
    """
    df = df.sort_values('date')
    
    # Convert date to datetime if needed
    if 'date' in df.columns and df['date'].dtype == object:
        df['date'] = pd.to_datetime(df['date'])
    
    # Set date as index if it's not already
    if 'date' in df.columns:
        df.set_index('date', inplace=True)
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df.index.dayofweek
    
    # Month
    df['month'] = df.index.month
    
    # Weekend indicator
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Day of year
    df['day_of_year'] = df.index.dayofyear
    
    # Quarter
    df['quarter'] = df.index.quarter
    
    return df


def create_features(df, lags=[1, 7, 14, 28], windows=[7, 14, 28]):
    """
    Create all features for time series forecasting.
    
    Args:
        df (pd.DataFrame): Raw store sales data
        lags (list): List of lag periods
        windows (list): List of window sizes
    
    Returns:
        pd.DataFrame: DataFrame with all features created
    """
    df = df.sort_values('date')
    
    # Add lag features (before date becomes index)
    df = add_lag_features(df, lags)
    
    # Add rolling features (before date becomes index)
    df = add_rolling_features(df, windows)
    
    # Add calendar features last (sets date as index)
    df = add_calendar_features(df)
    
    # Handle missing values caused by lag and rolling operations
    df.dropna(inplace=True)
    
    return df