import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pathlib
from typing import Tuple, Dict

def find_change_point(df: pd.DataFrame) -> Dict:
    """Find the most likely point of mean shift using multiple methods."""
    # Initialize storage for test statistics
    n_points = len(df)
    t_statistics = []
    p_values = []
    effect_sizes = []
    dates = []
    
    # Test each potential change point
    for i in range(12, n_points - 12):  # Skip first and last year to ensure enough data
        # Split data at this point
        before = df['mean_probability'].iloc[:i]
        after = df['mean_probability'].iloc[i:]
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(before, after)
        
        # Calculate effect size
        pooled_std = np.sqrt((before.var() + after.var()) / 2)
        cohens_d = (after.mean() - before.mean()) / pooled_std
        
        # Store results
        t_statistics.append(abs(t_stat))  # Use absolute value for comparison
        p_values.append(p_val)
        effect_sizes.append(abs(cohens_d))
        dates.append(df['date'].iloc[i])
    
    # Find the point with strongest evidence of change
    max_t_idx = np.argmax(t_statistics)
    max_effect_idx = np.argmax(effect_sizes)
    min_p_idx = np.argmin(p_values)
    
    # Calculate statistics for best change point
    best_idx = max_t_idx  # Using t-statistic as primary criterion
    change_date = dates[best_idx]
    before = df['mean_probability'].iloc[:best_idx + 12]  # +12 to convert from index in shortened list
    after = df['mean_probability'].iloc[best_idx + 12:]
    
    percent_change = ((after.mean() - before.mean()) / before.mean()) * 100
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    plt.plot(df['date'], df['mean_probability'], 'b-', label='Mean Probability')
    plt.axvline(x=change_date, color='r', linestyle='--', label='Most Likely Change Point')
    plt.title('Time Series with Detected Change Point')
    plt.xlabel('Date')
    plt.ylabel('Mean Probability')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    pathlib.Path('model_plots').mkdir(exist_ok=True)
    plt.savefig('model_plots/change_point_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'change_point_date': change_date,
        'before_mean': before.mean(),
        'after_mean': after.mean(),
        'before_std': before.std(),
        'after_std': after.std(),
        'percent_change': percent_change,
        't_statistic': t_statistics[best_idx],
        'p_value': p_values[best_idx],
        'effect_size': effect_sizes[best_idx],
        'alternative_dates': {
            'max_effect_date': dates[max_effect_idx],
            'min_p_date': dates[min_p_idx]
        }
    }

def main():
    try:
        df = pd.read_csv('model_plots/model_analysis_results.csv')
    except FileNotFoundError:
        print("Error: Could not find model_analysis_results.csv in model_plots directory")
        return
        
    results = find_change_point(df)
    
    print("\nChange Point Analysis")
    print("====================")
    print(f"\nMost Likely Change Point: {results['change_point_date']}")
    
    print(f"\nBefore Change Point:")
    print(f"Mean: {results['before_mean']:.6f}")
    print(f"Std Dev: {results['before_std']:.6f}")
    
    print(f"\nAfter Change Point:")
    print(f"Mean: {results['after_mean']:.6f}")
    print(f"Std Dev: {results['after_std']:.6f}")
    
    print(f"\nPercent Change: {results['percent_change']:.2f}%")
    
    print("\nStatistical Evidence:")
    print(f"t-statistic: {results['t_statistic']:.4f}")
    print(f"p-value: {results['p_value']:.10f}")
    print(f"Effect Size (Cohen's d): {results['effect_size']:.4f}")
    
    print("\nAlternative Change Points:")
    print(f"Maximum Effect Size: {results['alternative_dates']['max_effect_date']}")
    print(f"Minimum p-value: {results['alternative_dates']['min_p_date']}")
    
    # Additional interpretation
    print("\nInterpretation:")
    if results['p_value'] < 0.05:
        print("- The change point represents a statistically significant shift (p < 0.05)")
    if abs(results['effect_size']) > 0.8:
        print("- The magnitude of change is large (Cohen's d > 0.8)")

if __name__ == "__main__":
    main()