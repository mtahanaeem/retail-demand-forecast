import pandas as pd


def resample_data(df, store_id, product_family, freq='D'):
    """
    Resample store sales data to daily frequency.
    
    Args:
        df (pd.DataFrame): Raw store sales data
        store_id (int): Store identifier
        product_family (str): Product family name
        freq (str): Resampling frequency ('D' for daily, 'W' for weekly)
    
    Returns:
        pd.Series: Resampled time series of sales
    """
    store_data = df[
        (df['store_id'] == store_id) &
        (df['product_family'] == product_family)
    ].copy()
    
    store_data['date'] = pd.to_datetime(store_data['date'])
    store_data.set_index('date', inplace=True)
    
    daily_sales = store_data['sales'].resample(freq).sum()
    
    return daily_sales


def merge_calendar_data(df, calendar_path):
    """
    Merge store sales data with holiday and promotion calendar data.
    
    Args:
        df (pd.DataFrame): Store sales data
        calendar_path (str): Path to calendar data CSV file
    
    Returns:
        pd.DataFrame: Merged store sales with calendar features
    """
    calendar_df = pd.read_csv(calendar_path)
    calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    
    df_reset = df.reset_index()
    merged_df = pd.merge(df_reset, calendar_df, on='date', how='left')
    
    merged_df['is_holiday'] = merged_df['is_holiday'].fillna(False)
    merged_df['promotion'] = merged_df['promotion'].fillna(0)
    
    return merged_df


def get_unique_store_product_combinations(df):
    """
    Get all unique store and product family combinations for iteration.
    
    Args:
        df (pd.DataFrame): Raw store sales data
    
    Returns:
        list: List of (store_id, product_family) tuples
    """
    combinations = df[
        ['store_id', 'product_family']
    ].drop_duplicates().itertuples(index=False)
    
    return list(combinations)