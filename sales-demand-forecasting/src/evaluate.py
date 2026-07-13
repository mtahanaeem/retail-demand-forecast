import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_forecasts(actual, **model_forecasts):
    """
    Evaluate forecast performance for multiple models.
    
    Args:
        actual (np.ndarray): Actual values
        **model_forecasts: Dictionary of model name -> forecast array
    
    Returns:
        dict: Dictionary containing MAPE, RMSE, and MAE for each model
    """
    results = {}
    
    for model_name, forecast in model_forecasts.items():
        if len(forecast) != len(actual):
            raise ValueError(f"Length mismatch for {model_name}: "
                           f"actual={len(actual)}, forecast={len(forecast)}")
        
        # Handle potential NaN values
        mask = ~np.isnan(forecast) & ~np.isnan(actual)
        if mask.any():
            actual_valid = actual[mask]
            forecast_valid = forecast[mask]
        else:
            actual_valid = actual
            forecast_valid = forecast
        
        # Calculate metrics
        mape = np.mean(np.abs((actual_valid - forecast_valid) / actual_valid)) * 100
        rmse = np.sqrt(np.mean((actual_valid - forecast_valid) ** 2))
        mae = np.mean(np.abs(actual_valid - forecast_valid))
        
        results[model_name] = {
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae,
            'Actual': actual,
            'Forecast': forecast
        }
    
    return results


def plot_forecasts(actual, forecast_dict, title="Forecast Evaluation", save_path=None):
    """
    Plot actual vs forecasted values for multiple models.
    
    Args:
        actual (np.ndarray): Actual values
        forecast_dict (dict): Dictionary of model name -> forecast array
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(14, 8))
    
    # Plot actual values
    time_idx = np.arange(len(actual))
    plt.plot(time_idx, actual, 'k-', linewidth=2, label='Actual', alpha=0.8)
    
    # Plot forecasts for each model
    colors = ['red', 'blue', 'green', 'orange']
    for i, (model_name, forecast) in enumerate(forecast_dict.items()):
        if len(forecast) == len(actual):
            color = colors[i % len(colors)]
            plt.plot(time_idx, forecast, color=color, linewidth=1.5, 
                    label=f'{model_name} Forecast', alpha=0.8)
            
            # Highlight points with large errors (> 20% error)
            error_pct = np.abs((actual - forecast) / actual) * 100
            large_error_idx = np.where(error_pct > 20)[0]
            if len(large_error_idx) > 0:
                plt.scatter(large_error_idx, forecast[large_error_idx], 
                           color='red', s=80, alpha=0.6, zorder=10,
                           edgecolors='black', linewidths=1)
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def generate_comparison_table(results):
    """
    Generate a comparison table of model performance.
    
    Args:
        results (dict): Dictionary containing evaluation results
    
    Returns:
        pd.DataFrame: Comparison table with metrics
    """
    table_data = []
    
    for model_name, metrics in results.items():
        if isinstance(metrics, dict) and 'MAPE' in metrics:
            table_data.append({
                'Model': model_name,
                'MAPE (%)': metrics['MAPE'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE']
            })
    
    df = pd.DataFrame(table_data)
    
    # Sort by MAPE (best performer first)
    df = df.sort_values('MAPE (%)')
    df.reset_index(drop=True, inplace=True)
    
    return df


def print_summary_report(results):
    """
    Print a summary report of model evaluation.
    
    Args:
        results (dict): Dictionary containing evaluation results
    """
    print("=" * 60)
    print("FORECAST MODEL EVALUATION SUMMARY REPORT")
    print("=" * 60)
    
    df = generate_comparison_table(results)
    
    print("\nModel Performance Comparison (sorted by MAPE):")
    print("-" * 80)
    print(df.to_string(index=False))
    print("-" * 80)
    
    best_model = df.iloc[0]
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model['Model']}")
    print(f"   MAPE: {best_model['MAPE (%)']:.2f}%")
    print(f"   RMSE: {best_model['RMSE']:.2f}")
    print(f"   MAE: {best_model['MAE']:.2f}")
    
    print("\n📊 METRICS EXPLANATION:")
    print("   MAPE (%): Mean Absolute Percentage Error - lower is better")
    print("   RMSE: Root Mean Squared Error - lower is better")
    print("   MAE: Mean Absolute Error - lower is better")
    
    print("=" * 60)
