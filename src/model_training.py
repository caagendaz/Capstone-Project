"""
Model Training Module for E. coli Antibiotic Resistance Project

Trains and optimizes three models:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost

Includes hyperparameter optimization and cross-validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
import logging
import joblib
import json
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains and evaluates machine learning models for antibiotic resistance prediction.
    """
    
    def __init__(self, 
                 model_dir: str = "models",
                 results_dir: str = "results"):
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def prepare_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                    scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare and optionally scale feature data.
        """
        if scale:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        else:
            return X_train.values, X_test.values
    
    def train_logistic_regression(self, 
                                 X_train: np.ndarray, 
                                 y_train: pd.Series,
                                 cv: int = 5) -> LogisticRegression:
        """
        Train Logistic Regression model (baseline).
        
        Good for:
        - Linear relationships
        - Interpretability
        - Baseline performance
        """
        logger.info("Training Logistic Regression...")
        
        # Define hyperparameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000],
            'class_weight': ['balanced', None]
        }
        
        # Base model
        lr = LogisticRegression(random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            lr,
            param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        self.models['logistic_regression'] = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def train_random_forest(self,
                          X_train: np.ndarray,
                          y_train: pd.Series,
                          cv: int = 5,
                          n_iter: int = 50) -> RandomForestClassifier:
        """
        Train Random Forest model.
        
        Good for:
        - Non-linear relationships
        - Feature interactions
        - High-dimensional data
        - Feature importance
        """
        logger.info("Training Random Forest...")
        
        # Define hyperparameter distribution for random search
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
        
        # Base model
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Randomized search with cross-validation
        random_search = RandomizedSearchCV(
            rf,
            param_dist,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best CV ROC-AUC: {random_search.best_score_:.4f}")
        
        self.models['random_forest'] = random_search.best_estimator_
        return random_search.best_estimator_
    
    def train_xgboost(self,
                     X_train: np.ndarray,
                     y_train: pd.Series,
                     cv: int = 5,
                     n_iter: int = 50) -> xgb.XGBClassifier:
        """
        Train XGBoost model.
        
        Good for:
        - Complex non-linear patterns
        - High-dimensional genomic data
        - Handling imbalanced classes
        - State-of-the-art performance
        """
        logger.info("Training XGBoost...")
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Define hyperparameter distribution
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'scale_pos_weight': [1, scale_pos_weight]
        }
        
        # Base model
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        # Randomized search with cross-validation
        random_search = RandomizedSearchCV(
            xgb_model,
            param_dist,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best CV ROC-AUC: {random_search.best_score_:.4f}")
        
        self.models['xgboost'] = random_search.best_estimator_
        return random_search.best_estimator_
    
    def evaluate_model(self,
                      model: Any,
                      X_test: np.ndarray,
                      y_test: pd.Series,
                      model_name: str) -> Dict[str, float]:
        """
        Evaluate model performance on test set.
        """
        logger.info(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log results
        logger.info(f"\n{model_name} Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        # Classification report
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist()
        }
        
        return metrics
    
    def cross_validate_model(self,
                           model: Any,
                           X: np.ndarray,
                           y: pd.Series,
                           cv: int = 5) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation and return scores for multiple metrics.
        """
        logger.info("Performing cross-validation...")
        
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = {}
        
        for metric in scoring:
            scores = cross_val_score(model, X, y, cv=cv_splitter, 
                                    scoring=metric, n_jobs=-1)
            cv_results[metric] = scores
            logger.info(f"{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def plot_confusion_matrix(self, y_test: pd.Series, y_pred: np.ndarray,
                            model_name: str):
        """
        Plot confusion matrix.
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {model_name}')
        
        output_path = self.results_dir / f"confusion_matrix_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_path}")
        plt.close()
    
    def plot_roc_curves(self, X_test: np.ndarray, y_test: pd.Series):
        """
        Plot ROC curves for all trained models.
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{model_name} (AUC = {auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        output_path = self.results_dir / "roc_curves_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curves to {output_path}")
        plt.close()
    
    def plot_feature_importance(self, model: Any, feature_names: List[str],
                               model_name: str, top_n: int = 20):
        """
        Plot feature importance for tree-based models.
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"{model_name} does not have feature_importances_")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.tight_layout()
        
        output_path = self.results_dir / f"feature_importance_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance to {output_path}")
        plt.close()
    
    def compare_models(self) -> pd.DataFrame:
        """
        Create comparison table of model performances.
        """
        comparison = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            metrics['model'] = model_name
            comparison.append(metrics)
        
        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.set_index('model')
        
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        logger.info(f"\n{df_comparison.to_string()}")
        
        # Save to CSV
        output_path = self.results_dir / "metrics" / "model_comparison.csv"
        df_comparison.to_csv(output_path)
        logger.info(f"\nSaved comparison to {output_path}")
        
        return df_comparison
    
    def save_model(self, model: Any, model_name: str, antibiotic: str):
        """
        Save trained model to disk.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{antibiotic}_{timestamp}.pkl"
        filepath = self.model_dir / filename
        
        joblib.dump(model, filepath)
        logger.info(f"Saved model to {filepath}")
        
        # Also save scaler
        scaler_path = self.model_dir / f"scaler_{antibiotic}_{timestamp}.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        return filepath
    
    def save_results(self, antibiotic: str):
        """
        Save all results to JSON file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{antibiotic}_{timestamp}.json"
        filepath = self.results_dir / "metrics" / filename
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for model_name, result in self.results.items():
            results_serializable[model_name] = {
                'metrics': result['metrics'],
                'confusion_matrix': result['confusion_matrix']
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=4)
        
        logger.info(f"Saved results to {filepath}")
        return filepath


def train_all_models(X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_train: pd.Series,
                    y_test: pd.Series,
                    antibiotic: str,
                    cv: int = 5) -> ModelTrainer:
    """
    Train all models and return trainer with results.
    """
    trainer = ModelTrainer()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING MODELS FOR {antibiotic}")
    logger.info(f"{'='*60}\n")
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    logger.info(f"Number of features: {X_train.shape[1]}")
    logger.info(f"Class distribution (train): {y_train.value_counts().to_dict()}")
    
    # Prepare data
    X_train_scaled, X_test_scaled = trainer.prepare_data(X_train, X_test, scale=True)
    
    # Train models
    models_to_train = {
        'logistic_regression': trainer.train_logistic_regression,
        'random_forest': trainer.train_random_forest,
        'xgboost': trainer.train_xgboost
    }
    
    for model_name, train_func in models_to_train.items():
        try:
            # Train
            model = train_func(X_train_scaled, y_train, cv=cv)
            
            # Evaluate
            trainer.evaluate_model(model, X_test_scaled, y_test, model_name)
            
            # Plot confusion matrix
            y_pred = model.predict(X_test_scaled)
            trainer.plot_confusion_matrix(y_test, y_pred, model_name)
            
            # Save model
            trainer.save_model(model, model_name, antibiotic)
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
    
    # Compare models
    trainer.compare_models()
    
    # Plot ROC curves
    trainer.plot_roc_curves(X_test_scaled, y_test)
    
    # Plot feature importance for tree models
    if 'random_forest' in trainer.models:
        trainer.plot_feature_importance(
            trainer.models['random_forest'],
            X_train.columns.tolist(),
            'random_forest'
        )
    
    if 'xgboost' in trainer.models:
        trainer.plot_feature_importance(
            trainer.models['xgboost'],
            X_train.columns.tolist(),
            'xgboost'
        )
    
    # Save results
    trainer.save_results(antibiotic)
    
    return trainer


def main():
    """
    Main training pipeline.
    """
    # Example usage
    logger.info("Model training pipeline ready")
    
    # Load preprocessed data
    # from data_preprocessing import EcoliDataPreprocessor
    # preprocessor = EcoliDataPreprocessor()
    # df = preprocessor.load_data("data/processed/merged_data_CLSI.csv")
    # 
    # # Train for specific antibiotic
    # antibiotic = "CIPROFLOXACIN"
    # X_train, X_test, y_train, y_test = preprocessor.prepare_modeling_data(
    #     df, antibiotic, test_size=0.2, random_state=42
    # )
    # 
    # # Train all models
    # trainer = train_all_models(X_train, X_test, y_train, y_test, 
    #                           antibiotic, cv=5)


if __name__ == "__main__":
    main()
