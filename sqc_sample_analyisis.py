import numpy as np
from scipy import stats
import math

def analyze_sample_size(n_sample=84, n_population=24000, confidence_level=0.95):
    """Calculate margin of error and related statistics for our sample."""
    # Calculate margin of error for proportions
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    p = 0.5  # Use 0.5 for maximum margin of error
    
    # Finite population correction
    fpc = math.sqrt((n_population - n_sample) / (n_population - 1))
    margin_of_error = z_score * math.sqrt((p * (1-p)) / n_sample) * fpc
    
    # For mean estimation (using our observed std from the data)
    std_prob = 0.003  # Approximate std of mean_probability from our data
    margin_of_error_mean = z_score * (std_prob / math.sqrt(n_sample)) * fpc
    
    # Calculate required sample size for different margins of error
    def required_sample_size(desired_margin):
        n = (z_score**2 * p * (1-p)) / (desired_margin**2)
        # Finite population correction adjustment
        n = (n * n_population) / (n + n_population - 1)
        return math.ceil(n)
    
    return {
        'current_margin_of_error': margin_of_error,
        'mean_margin_of_error': margin_of_error_mean,
        'sample_size_1pct': required_sample_size(0.01),
        'sample_size_5pct': required_sample_size(0.05),
        'sample_size_10pct': required_sample_size(0.10),
        'population_fraction': n_sample / n_population
    }

# Run analysis
results = analyze_sample_size()

print("\nSample Size Analysis")
print("===================")
print(f"Monthly sample size: 84 papers")
print(f"Monthly population size: 24,000 papers")
print(f"Sample fraction: {results['population_fraction']*100:.2f}%")

print("\nMargin of Error Analysis:")
print(f"General margin of error: ±{results['current_margin_of_error']*100:.2f}%")
print(f"Margin of error for mean probability: ±{results['mean_margin_of_error']*100:.4f}%")

print("\nRequired sample sizes for different margins of error:")
print(f"For ±1% margin of error: {results['sample_size_1pct']} papers")
print(f"For ±5% margin of error: {results['sample_size_5pct']} papers")
print(f"For ±10% margin of error: {results['sample_size_10pct']} papers")