import json
import pathlib
from typing import Dict, List, Tuple
import numpy as np
from bino_true import Binoculars
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from sklearn.linear_model import LogisticRegression

class ModelArxivAnalyzer:
    def __init__(self, data_dir: str = "arxiv_samples", model_path: str = "model_results/binoculars_classifier.joblib"):
        self.data_dir = pathlib.Path(data_dir)
        self.binoculars = Binoculars(mode="accuracy")
        # Load the trained logistic regression model
        self.model = joblib.load(model_path)
        
    def load_papers(self) -> Dict[int, List[dict]]:
        """Load papers from all year files."""
        papers_by_year = {}
        for file in self.data_dir.glob("papers_*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                year = data['year']
                papers_by_year[year] = data['papers']
        return papers_by_year
        
    def analyze_papers(self, papers_by_year: Dict[int, List[dict]]) -> pd.DataFrame:
        """Analyze papers by month using Binoculars and logistic regression model."""
        results = []
        
        for year, papers in papers_by_year.items():
            print(f"\nAnalyzing papers from {year}...")
            
            # Group papers by month
            papers_by_month = {month: [] for month in range(1, 13)}
            for paper in papers:
                month = int(paper['created'][5:7])
                papers_by_month[month].append(paper)
            
            # Process each month
            for month in range(1, 13):
                month_papers = papers_by_month[month]
                scores = []
                probabilities = []
                predictions = []
                
                print(f"Processing {year}-{month:02d} ({len(month_papers)} papers)...")
                for paper in tqdm(month_papers):
                    try:
                        # Get Binoculars score
                        score = self.binoculars.compute_score(paper['abstract'])
                        if isinstance(score, list):
                            score = score[0]
                            
                        if np.isfinite(score):
                            scores.append(score)
                            # Get model prediction and probability
                            prob = self.model.predict_proba([[score]])[0][1]  # Probability of AI class
                            pred = self.model.predict([[score]])[0]
                            probabilities.append(prob)
                            predictions.append(pred)
                            
                    except Exception as e:
                        print(f"Error processing paper: {e}")
                        continue
                
                if not scores:
                    print(f"No valid scores for {year}-{month:02d}")
                    continue
                    
                scores = np.array(scores)
                probabilities = np.array(probabilities)
                predictions = np.array(predictions)
                
                # Calculate statistics
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                score_ci = 1.96 * std_score / np.sqrt(len(scores))
                
                # Calculate model-based statistics
                mean_prob = np.mean(probabilities)
                prob_std = np.std(probabilities)
                prob_ci = 1.96 * prob_std / np.sqrt(len(probabilities))
                
                # Calculate AI detection percentage based on model predictions
                ai_percentage = 100 * np.mean(predictions)
                ai_percentage_std = 100 * np.std(predictions) / np.sqrt(len(predictions))
                
                results.append({
                    'year': year,
                    'month': month,
                    'date': f"{year}-{month:02d}",
                    'mean_score': mean_score,
                    'score_ci': score_ci,
                    'mean_probability': mean_prob,
                    'probability_ci': prob_ci,
                    'ai_percentage': ai_percentage,
                    'ai_percentage_ci': ai_percentage_std,
                    'n_papers': len(scores)
                })
        
        return pd.DataFrame(results).sort_values(['year', 'month'])
        
    def plot_results(self, df: pd.DataFrame, output_dir: str = "model_plots"):
        """Create plots of the model-based analysis results."""
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Set basic style parameters
        plt.rcParams['figure.figsize'] = (15, 6)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
        # Color scheme
        years = df['year'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
        
        # Plot mean probabilities (bar plot)
        plt.figure()
        bars = plt.bar(df['date'], df['mean_probability'], 
                    yerr=df['probability_ci'],
                    capsize=3)
        
        for year, color in zip(years, colors):
            year_mask = df['year'] == year
            for bar in np.array(bars)[year_mask]:
                bar.set_color(color)
        
        plt.title('Average AI Probability Over Time (Model-Based)')
        plt.xlabel('Date')
        plt.ylabel('Probability')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'average_probability_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot mean probabilities (time series)
        plt.figure()
        plt.errorbar(range(len(df)), df['mean_probability'], 
                    yerr=df['probability_ci'],
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    markersize=6,
                    capsize=3)
        
        plt.title('AI Probability Time Series (Model-Based)')
        plt.xlabel('Time Period')
        plt.ylabel('Probability')
        num_ticks = 8
        tick_positions = np.linspace(0, len(df)-1, num_ticks, dtype=int)
        plt.xticks(tick_positions, df['date'].iloc[tick_positions], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'average_probability_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot AI percentages based on model predictions
        plt.figure()
        bars = plt.bar(df['date'], df['ai_percentage'],
                    yerr=df['ai_percentage_ci'],
                    capsize=3)
        
        for year, color in zip(years, colors):
            year_mask = df['year'] == year
            for bar in np.array(bars)[year_mask]:
                bar.set_color(color)
        
        plt.title('Percentage of AI-Generated Abstracts (Model Predictions)')
        plt.xlabel('Date')
        plt.ylabel('Percentage (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'ai_percentage_model_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save results to CSV
        df.to_csv(output_dir / 'model_analysis_results.csv', index=False)
        return output_dir

def main():
    analyzer = ModelArxivAnalyzer()
    
    print("Loading papers...")
    papers_by_year = analyzer.load_papers()
    
    print("Analyzing papers with logistic regression model...")
    results_df = analyzer.analyze_papers(papers_by_year)
    
    print("Creating plots...")
    output_dir = analyzer.plot_results(results_df)
    
    print(f"\nResults saved to {output_dir}")
    print("\nSummary of results:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()