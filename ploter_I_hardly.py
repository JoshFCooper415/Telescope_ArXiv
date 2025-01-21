import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Tuple

class TimeSeriesSmoother:
    def __init__(self, results_path: str = "plot2"):
        """
        Initialize the smoother with the path to analysis results.
        
        Args:
            results_path: Path to the CSV file containing analysis results
        """
        results_path = results_path + "/analysis_results.csv"
        self.df = pd.read_csv(results_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
    def kalman_filter(self, data: np.ndarray, process_variance: float = 1e-4, 
                     measurement_variance: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Kalman filter to the data series.
        
        Args:
            data: Input time series data
            process_variance: How fast the true state can change (Q)
            measurement_variance: How noisy the measurements are (R)
            
        Returns:
            Tuple of (filtered_states, state_covariances)
        """
        n = len(data)
        x_hat = data[0]
        P = 1.0
        
        filtered_states = np.zeros(n)
        state_covariances = np.zeros(n)
        
        for t in range(n):
            # Predict
            x_hat_minus = x_hat
            P_minus = P + process_variance
            
            # Update
            K = P_minus / (P_minus + measurement_variance)
            x_hat = x_hat_minus + K * (data[t] - x_hat_minus)
            P = (1 - K) * P_minus
            
            # Store
            filtered_states[t] = x_hat
            state_covariances[t] = P
            
        return filtered_states, np.sqrt(state_covariances)

    def plot_simple_kalman(self, output_dir: str = "plot2"):
        """
        Create simplified plots with only Kalman smoothing and ChatGPT release point.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        plt.rcParams['figure.figsize'] = (15, 8)
        chatgpt_release = pd.to_datetime('2022-11-30')
        
        metrics = [
            ('mean_score', 'Average Telescope Score'),
            ('ai_percentage', 'AI-Generated Abstract Percentage')
        ]
        
        for column, title in metrics:
            fig, ax = plt.subplots()
            
            # Calculate Kalman filtered data
            filtered_states, uncertainties = self.kalman_filter(
                self.df[column].values, 
                process_variance=1e-4,  # Using the smoother configuration
                measurement_variance=1e-2
            )
            
            # Plot Kalman smoothed line
            ax.plot(self.df['date'], filtered_states,
                   color='#2ecc71',  # Nice green color
                   linewidth=2.5,
                   label='Kalman Smoothed')
            
            # Add confidence interval
            ax.fill_between(self.df['date'],
                          filtered_states - uncertainties,
                          filtered_states + uncertainties,
                          color='#2ecc71',
                          alpha=0.2)
            
            # Add ChatGPT release point
            if chatgpt_release >= min(self.df['date']) and chatgpt_release <= max(self.df['date']):
                # Find the closest date and corresponding value
                closest_idx = np.abs(self.df['date'] - chatgpt_release).argmin()
                chatgpt_value = filtered_states[closest_idx]
                
                # Plot point and annotation
                ax.scatter(chatgpt_release, chatgpt_value, 
                         color='red', s=100, zorder=5,
                         label='ChatGPT Release')
                
                # Add annotation with arrow
                ax.annotate('ChatGPT Release',
                          xy=(chatgpt_release, chatgpt_value),
                          xytext=(20, 20), textcoords='offset points',
                          color='red',
                          fontweight='bold',
                          arrowprops=dict(arrowstyle='->',
                                        color='red',
                                        alpha=0.7))
            
            # Customize plot
            ax.set_title(title, fontsize=14, pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Rotate x-axis dates for better readability
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_dir / f'{column}_kalman_simple.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

def main():
    path = "plot_gemma"
    smoother = TimeSeriesSmoother(path)
    smoother.plot_simple_kalman(path)
    print("Simple Kalman smoothed plots have been generated in the output directory.")

if __name__ == "__main__":
    main()