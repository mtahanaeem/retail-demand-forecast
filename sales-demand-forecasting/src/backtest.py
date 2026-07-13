import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class BacktestSplitter:
    """Implement walk-forward validation for time series."""

    def __init__(self, train_size=None, test_size=1, step=1, min_history=30):
        """
        Initialize walk-forward backtest splitter.
        
        Args:
            train_size (int): Number of training samples (auto-determined if None)
            test_size (int): Number of test samples (default: 1)
            step (int): Step size for moving window (default: 1)
            min_history (int): Minimum history required for training (default: 30)
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step = step
        self.min_history = min_history
        self.splits = []

    def get_splits(self, series):
        """
        Generate time-ordered walk-forward splits.
        
        Args:
            series (pd.Series): Time series data
        
        Returns:
            list: List of (train_idx, test_idx) tuples
        """
        splits = []
        n = len(series)
        
        if self.train_size is None:
            self.train_size = n - self.test_size - self.min_history
        
        start = self.min_history
        end = n - self.test_size
        
        for i in range(start, end, self.step):
            train_end = i
            train_start = max(0, train_end - self.train_size)
            test_start = train_end
            test_end = test_start + self.test_size
            
            splits.append((train_start, train_end, test_start, test_end))
        
        self.splits = splits
        return splits

    def split(self, series):
        """
        Generate train/test splits for walk-forward validation.
        
        Args:
            series (pd.Series): Time series data
        
        Returns:
            generator: Generator yielding (train_series, test_series) tuples
        """
        if not self.splits:
            self.get_splits(series)
        
        for train_start, train_end, test_start, test_end in self.splits:
            train = series.iloc[train_start:train_end]
            test = series.iloc[test_start:test_end]
            yield train, test

    def visualize_splits(self, series, save_path=None):
        """
        Visualize the walk-forward split pattern.
        
        Args:
            series (pd.Series): Time series data
            save_path (str): Path to save visualization
        """
        self.get_splits(series)
        
        plt.figure(figsize=(14, 6))
        plt.plot(series.index, series.values, label='Time Series', alpha=0.7)
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(self.splits[:10]):  # Show first 10 splits
            color = 'green' if i % 2 == 0 else 'orange'
            plt.axvspan(train_start, train_end, alpha=0.2, color=color, label='Train' if i == 0 else "")
            plt.axvspan(test_start, test_end, alpha=0.3, color='red', label='Test' if i == 0 else "")
        
        plt.xlabel('Time Index')
        plt.ylabel('Value')
        plt.title('Walk-Forward Backtest Splits')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class TimeSeriesBacktester:
    """Perform walk-forward backtesting for forecasting models."""

    def __init__(self, n_test=12, model_class=None, **model_kwargs):
        """
        Initialize backtester.
        
        Args:
            n_test (int): Number of periods to forecast in each test (default: 12)
            model_class (class): Model class to use for testing
            **model_kwargs: Additional arguments for model initialization
        """
        self.n_test = n_test
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.results = {}
        self.histories = {}
        self.forecasts = {}

    def backtest_model(self, series, evaluate_func=None):
        """
        Perform walk-forward backtesting for a model.
        
        Args:
            series (pd.Series): Time series data
            evaluate_func (function): Function to evaluate forecasts
        
        Returns:
            dict: Dictionary containing actual and forecasted values
        """
        if self.model_class is None:
            raise ValueError("Model class must be specified")
        
        y_test = series[-self.n_test:].values
        history = series[:-self.n_test].tolist()
        forecasts = []
        
        for t in range(len(y_test)):
            model = self.model_class(**self.model_kwargs)
            
            if hasattr(model, 'prepare_features'):
                X = pd.DataFrame({'sales': history})
                X = model.prepare_features(X)
                model.fit(X, pd.Series(history))
            elif hasattr(model, 'fit'):
                model.fit(pd.Series(history))
            
            forecast = model.forecast(pd.DataFrame({'sales': history}))
            if len(forecast) > 0:
                forecasts.append(forecast[0])
            else:
                forecasts.append(history[-1])
            
            history.append(y_test[t])
        
        result = {
            'actual': y_test,
            'forecast': np.array(forecasts)
        }
        
        if evaluate_func:
            result['metrics'] = evaluate_func(y_test, np.array(forecasts))
        
        return result