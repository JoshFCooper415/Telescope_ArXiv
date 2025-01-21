import json
import pathlib
from typing import Dict, List, Tuple
import numpy as np
from bino_true import Binoculars
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class BinocularsTrainer:
    def __init__(self, csv_path: str = "4o_essay_set.csv"):
        self.csv_path = pathlib.Path(csv_path)
        self.binoculars = Binoculars(mode="accuracy")
        
    def load_data(self) -> Tuple[List[str], List[int]]:
        """Load both original and AI-rewritten texts from CSV."""
        # Load the CSV file
        df = pd.read_csv(self.csv_path)
        
        # Extract texts and create labels
        texts = []
        labels = []
        
        # Add original texts (label 0)
        print("Processing original texts...")
        original_texts = df['original_text'].dropna()
        texts.extend(original_texts)
        labels.extend([0] * len(original_texts))
            
        # Add rewritten/AI texts (label 1)
        print("Processing AI-rewritten texts...")
        rewritten_texts = df['rewritten_text'].dropna()
        texts.extend(rewritten_texts)
        labels.extend([1] * len(rewritten_texts))
            
        return texts, labels
    
    def compute_binoculars_scores(self, texts: List[str]) -> np.ndarray:
        """Compute Binoculars scores for all texts."""
        scores = []
        print("Computing Binoculars scores...")
        for text in tqdm(texts):
            try:
                score = self.binoculars.compute_score(text)
                if isinstance(score, list):
                    score = score[0]
                scores.append(score if np.isfinite(score) else np.nan)
            except Exception as e:
                print(f"Error processing text: {e}")
                scores.append(np.nan)
                
        return np.array(scores)
    
    def prepare_features(self, scores: np.ndarray) -> np.ndarray:
        """Prepare feature matrix for training."""
        return scores.reshape(-1, 1)
    
    def train_model(self, X: np.ndarray, y: np.array, 
                    test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[LogisticRegression, Dict]:
        """Train and evaluate logistic regression model."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        model = LogisticRegression(random_state=random_state)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        
        # Generate evaluation metrics
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_indices': (X_test, y_test, y_pred)  # Save for plotting
        }
        
        return model, metrics
    
    def plot_results(self, X: np.ndarray, metrics: Dict, output_dir: str = "model_results"):
        """Create visualization plots for the results."""
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], 
                   annot=True, 
                   fmt='d',
                   cmap='Blues',
                   xticklabels=['Human', 'AI'],
                   yticklabels=['Human', 'AI'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot score distributions
        X_test, y_test, y_pred = metrics['test_indices']
        plt.figure(figsize=(10, 6))
        for label, label_name in [(0, 'Human'), (1, 'AI')]:
            scores = X_test[y_test == label]
            plt.hist(scores, bins=50, alpha=0.5, label=label_name, density=True)
        plt.title('Distribution of Binoculars Scores')
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to JSON
        metrics_to_save = {
            'train_accuracy': metrics['train_accuracy'],
            'test_accuracy': metrics['test_accuracy'],
            'classification_report': metrics['classification_report']
        }
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_to_save, f, indent=2)

def main():
    trainer = BinocularsTrainer()
    
    # Load data
    print("Loading data...")
    texts, labels = trainer.load_data()
    
    # Compute scores
    scores = trainer.compute_binoculars_scores(texts)
    
    # Remove any samples with NaN scores
    valid_mask = ~np.isnan(scores)
    X = scores[valid_mask].reshape(-1, 1)
    y = np.array(labels)[valid_mask]
    
    # Train and evaluate model
    print("\nTraining model...")
    model, metrics = trainer.train_model(X, y)
    
    # Plot results
    print("\nGenerating plots...")
    trainer.plot_results(X, metrics)
    
    print("\nModel training completed!")
    print(f"Train accuracy: {metrics['train_accuracy']:.3f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    
    # Save the model
    import joblib
    joblib.dump(model, 'model_results/binoculars_classifier.joblib')
    print("\nModel saved to model_results/binoculars_classifier.joblib")

if __name__ == "__main__":
    main()