import json
import pathlib
from typing import Dict, List, Tuple
import numpy as np
from binoculars_detector import Binoculars
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

class BinocularsTrainer:
    def __init__(self, data_dir: str = "generated_samples", 
                 original_dir: str = "arxiv_samples",
                 sampling_strategy: str = "over"):
        """
        Initialize the trainer with sampling strategy.
        
        Args:
            data_dir: Directory containing generated samples
            original_dir: Directory containing original samples
            sampling_strategy: Type of sampling to use ("over", "under", or "none")
        """
        self.data_dir = pathlib.Path(data_dir)
        self.original_dir = pathlib.Path(original_dir)
        self.binoculars = Binoculars(mode="accuracy")
        self.sampling_strategy = sampling_strategy
        
    def load_data(self) -> Tuple[List[str], List[str], List[int]]:
        """Load both original and AI-generated abstracts."""
        # [Previous load_data implementation remains the same]
        generated_file = self.data_dir / "generated_abstracts_2021.json"
        with open(generated_file, 'r', encoding='utf-8') as f:
            generated_data = json.load(f)
            
        original_file = self.original_dir / "papers_2021.json"
        with open(original_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
            
        abstracts = []
        labels = []
        
        print("Processing original abstracts...")
        for paper in tqdm(original_data['papers']):
            abstracts.append(paper['abstract'])
            labels.append(0)
            
        print("Processing generated abstracts...")
        for paper in tqdm(generated_data['papers']):
            abstracts.append(paper['generated_abstract'])
            labels.append(1)
            
        return abstracts, labels
    
    def compute_binoculars_scores(self, abstracts: List[str]) -> np.ndarray:
        """Compute Binoculars scores for all abstracts."""
        # [Previous compute_binoculars_scores implementation remains the same]
        scores = []
        print("Computing Binoculars scores...")
        for abstract in tqdm(abstracts):
            try:
                score = self.binoculars.compute_score(abstract)
                if isinstance(score, list):
                    score = score[0]
                scores.append(score if np.isfinite(score) else np.nan)
            except Exception as e:
                print(f"Error processing abstract: {e}")
                scores.append(np.nan)
                
        return np.array(scores)
    
    def prepare_features(self, scores: np.ndarray) -> np.ndarray:
        """Prepare feature matrix for training."""
        return scores.reshape(-1, 1)
    
    def create_sampling_pipeline(self, random_state: int = 42):
        """Create a pipeline with the specified sampling strategy."""
        if self.sampling_strategy == "over":
            return Pipeline([
                ('sampler', SMOTE(random_state=random_state)),
                ('classifier', LogisticRegression(random_state=random_state))
            ])
        elif self.sampling_strategy == "under":
            return Pipeline([
                ('sampler', RandomUnderSampler(random_state=random_state)),
                ('classifier', LogisticRegression(random_state=random_state))
            ])
        else:  # "none"
            return LogisticRegression(random_state=random_state)
    
    def train_model(self, X: np.ndarray, y: np.array, 
                    test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[LogisticRegression, Dict]:
        """Train and evaluate model with sampling."""
        # Split data before sampling to prevent data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create and train model pipeline
        model = self.create_sampling_pipeline(random_state)
        
        # Fit the pipeline
        if self.sampling_strategy in ["over", "under"]:
            model.fit(X_train, y_train)
            final_model = model.named_steps['classifier']
        else:
            model.fit(X_train, y_train)
            final_model = model
        
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
            'test_indices': (X_test, y_test, y_pred)
        }
        
        return final_model, metrics
    
    def plot_results(self, X: np.ndarray, metrics: Dict, output_dir: str = "model_results"):
        """Create visualization plots for the results."""
        # [Previous plot_results implementation remains the same]
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
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
        
        metrics_to_save = {
            'train_accuracy': metrics['train_accuracy'],
            'test_accuracy': metrics['test_accuracy'],
            'classification_report': metrics['classification_report']
        }
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_to_save, f, indent=2)

def main():
    # Create trainer with specified sampling strategy
    trainer = BinocularsTrainer(sampling_strategy="under")  # or "under" for undersampling
    
    print("Loading data...")
    abstracts, labels = trainer.load_data()
    
    scores = trainer.compute_binoculars_scores(abstracts)
    
    valid_mask = ~np.isnan(scores)
    X = scores[valid_mask].reshape(-1, 1)
    y = np.array(labels)[valid_mask]
    
    # Print class distribution before sampling
    print("\nClass distribution before sampling:")
    print(pd.Series(y).value_counts(normalize=True))
    
    print("\nTraining model...")
    model, metrics = trainer.train_model(X, y)
    
    print("\nGenerating plots...")
    trainer.plot_results(X, metrics)
    
    print("\nModel training completed!")
    print(f"Train accuracy: {metrics['train_accuracy']:.3f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    
    import joblib
    joblib.dump(model, 'model_results/binoculars_classifier.joblib')
    print("\nModel saved to model_results/binoculars_classifier.joblib")

if __name__ == "__main__":
    main()