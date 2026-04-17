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
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import RFE, SelectFromModel
import xgboost as xgb

# Try to import SMOTE, install if not available
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

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
                 results_dir: str = "results",
                 figures_dir: Optional[str] = None,
                 metrics_dir: Optional[str] = None,
                 use_smote: bool = True,
                 use_feature_selection: bool = True,
                 n_features: int = 30,
                 artifact_prefix: str = ""):
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.metrics_dir = Path(metrics_dir) if metrics_dir else self.results_dir / "metrics"
        self.figures_dir = Path(figures_dir) if figures_dir else self.results_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        # Restricted Windows sandboxes can block loky/joblib subprocesses.
        # Use serial execution so training can still complete reliably.
        self.parallel_jobs = 1
        self.use_smote = use_smote and SMOTE_AVAILABLE
        self.use_feature_selection = use_feature_selection
        self.n_features = n_features
        self.artifact_prefix = "".join(
            char if char.isalnum() or char in ("_", "-") else "_"
            for char in artifact_prefix
        ).strip("_")

    def _artifact_name(self, base_name: str) -> str:
        """Attach a run-specific prefix to avoid overwriting artifacts."""
        return f"{base_name}_{self.artifact_prefix}" if self.artifact_prefix else base_name

    def _build_pipeline(self, model_name: str, model: Any, num_input_features: int):
        """
        Build a leakage-safe pipeline so scaling, SMOTE, and feature selection
        occur inside each cross-validation fold.
        """
        steps = []

        if model_name == 'logistic_regression':
            steps.append(('scaler', StandardScaler()))
        else:
            steps.append(('scaler', 'passthrough'))

        if self.use_smote:
            steps.append(('smote', SMOTE(random_state=42)))

        k_features = 'all'
        if self.use_feature_selection and num_input_features > self.n_features:
            k_features = min(self.n_features, num_input_features)
        steps.append(('select', SelectKBest(mutual_info_classif, k=k_features)))
        steps.append(('model', model))

        pipeline_class = ImbPipeline if self.use_smote else SkPipeline
        return pipeline_class(steps)

    def _unwrap_estimator_and_features(self, model: Any, feature_names: List[str]) -> Tuple[Any, List[str]]:
        """
        Extract the fitted estimator and surviving feature names from a pipeline.
        """
        final_model = model
        selected_features = list(feature_names)

        if hasattr(model, 'named_steps'):
            selector = model.named_steps.get('select')
            if selector is not None and selector != 'passthrough' and hasattr(selector, 'get_support'):
                support = selector.get_support()
                if len(support) == len(feature_names):
                    selected_features = [name for name, keep in zip(feature_names, support) if keep]

            final_model = model.named_steps.get('model', model)

        return final_model, selected_features
        
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
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['liblinear', 'saga'],
            'model__max_iter': [1000],
            'model__class_weight': ['balanced', None]
        }
        
        # Base model
        lr = self._build_pipeline(
            'logistic_regression',
            LogisticRegression(random_state=42),
            num_input_features=X_train.shape[1]
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            lr,
            param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=self.parallel_jobs,
            verbose=1,
            error_score='raise'
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
            'model__n_estimators': [100, 200, 300, 500],
            'model__max_depth': [10, 20, 30, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None],
            'model__bootstrap': [True, False],
            'model__class_weight': ['balanced', 'balanced_subsample', None]
        }
        
        # Base model
        rf = self._build_pipeline(
            'random_forest',
            RandomForestClassifier(random_state=42, n_jobs=self.parallel_jobs),
            num_input_features=X_train.shape[1]
        )
        
        # Randomized search with cross-validation
        random_search = RandomizedSearchCV(
            rf,
            param_dist,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=self.parallel_jobs,
            verbose=1,
            random_state=42,
            error_score='raise'
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
        scale_pos_weight_values = [1] if self.use_smote else [1, scale_pos_weight]
        
        # Define hyperparameter distribution
        param_dist = {
            'model__n_estimators': [100, 200, 300, 500],
            'model__max_depth': [3, 5, 7, 9],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__subsample': [0.6, 0.8, 1.0],
            'model__colsample_bytree': [0.6, 0.8, 1.0],
            'model__min_child_weight': [1, 3, 5],
            'model__gamma': [0, 0.1, 0.2],
            'model__scale_pos_weight': scale_pos_weight_values
        }
        
        # Base model
        xgb_model = self._build_pipeline(
            'xgboost',
            xgb.XGBClassifier(
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=self.parallel_jobs
            ),
            num_input_features=X_train.shape[1]
        )
        
        # Randomized search with cross-validation
        random_search = RandomizedSearchCV(
            xgb_model,
            param_dist,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=self.parallel_jobs,
            verbose=1,
            random_state=42,
            error_score='raise'
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
                                    scoring=metric, n_jobs=self.parallel_jobs)
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
        
        output_path = self.figures_dir / f"{self._artifact_name(f'confusion_matrix_{model_name}')}.png"
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
        
        output_path = self.figures_dir / f"{self._artifact_name('roc_curves_comparison')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curves to {output_path}")
        plt.close()
    
    def plot_feature_importance(self, model: Any, feature_names: List[str],
                               model_name: str, top_n: int = 20):
        """
        Plot feature importance for tree-based models.
        """
        fitted_model, selected_features = self._unwrap_estimator_and_features(model, feature_names)

        if not hasattr(fitted_model, 'feature_importances_'):
            logger.warning(f"{model_name} does not have feature_importances_")
            return
        
        importances = fitted_model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [selected_features[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.tight_layout()
        
        output_path = self.figures_dir / f"{self._artifact_name(f'feature_importance_{model_name}')}.png"
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
        if df_comparison.empty:
            raise RuntimeError("No model results were generated, so comparison could not be created.")
        df_comparison = df_comparison.set_index('model')
        
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        logger.info(f"\n{df_comparison.to_string()}")
        
        # Save to CSV
        output_path = self.metrics_dir / f"{self._artifact_name('model_comparison')}.csv"
        df_comparison.to_csv(output_path)
        logger.info(f"\nSaved comparison to {output_path}")
        
        return df_comparison
    
    def save_model(self, model: Any, model_name: str, antibiotic: str):
        """
        Save trained model to disk.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_antibiotic = self._artifact_name(antibiotic)
        filename = f"{model_name}_{safe_antibiotic}_{timestamp}.pkl"
        filepath = self.model_dir / filename
        
        joblib.dump(model, filepath)
        logger.info(f"Saved model to {filepath}")
        
        if hasattr(self.scaler, 'mean_'):
            scaler_path = self.model_dir / f"scaler_{safe_antibiotic}_{timestamp}.pkl"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
        else:
            logger.info("Scaler is embedded in the saved model pipeline")
        
        return filepath
    
    def save_results(self, antibiotic: str):
        """
        Save all results to JSON file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._artifact_name('results')}_{timestamp}.json"
        filepath = self.metrics_dir / filename
        
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

    def apply_smote(self, X_train: np.ndarray, y_train: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to handle class imbalance.
        """
        if not SMOTE_AVAILABLE:
            logger.warning("SMOTE not available. Returning original data.")
            return X_train, y_train.values if hasattr(y_train, 'values') else y_train
            
        logger.info("Applying SMOTE for class balancing...")
        logger.info(f"Before SMOTE: {pd.Series(y_train).value_counts().to_dict()}")
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
        return X_resampled, y_resampled
    
    def select_features_rfe(self, X_train: np.ndarray, y_train: np.ndarray,
                           n_features: int = 20) -> Tuple[np.ndarray, List[int]]:
        """
        Select top features using Recursive Feature Elimination.
        """
        logger.info(f"Selecting top {n_features} features using RFE...")
        
        # Use Random Forest as base estimator
        base_estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=self.parallel_jobs
        )
        
        n_features = min(n_features, X_train.shape[1])
        rfe = RFE(estimator=base_estimator, n_features_to_select=n_features, step=1)
        rfe.fit(X_train, y_train)
        
        selected_indices = list(np.where(rfe.support_)[0])
        logger.info(f"Selected {len(selected_indices)} features")
        
        self.feature_selector = rfe
        return X_train[:, selected_indices], selected_indices
    
    def find_optimal_threshold(self, model: Any, X_train: np.ndarray,
                               y_train: np.ndarray, cv: int = 5) -> float:
        """
        Find an operating threshold using out-of-fold predictions on the
        training data only, so the test set remains untouched.
        """
        logger.info("Finding optimal classification threshold from training-only out-of-fold predictions...")

        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        y_proba = cross_val_predict(
            model,
            X_train,
            y_train,
            cv=cv_splitter,
            method='predict_proba',
            n_jobs=self.parallel_jobs
        )[:, 1]
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_train, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
        return best_threshold
    
    def train_stacking_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                               cv: int = 5) -> StackingClassifier:
        """
        Create a stacking ensemble of all base models.
        """
        logger.info("Training stacking ensemble...")
        
        # Base estimators
        estimators = [
            ('lr', self._build_pipeline(
                'logistic_regression',
                LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
                num_input_features=X_train.shape[1]
            )),
            ('rf', self._build_pipeline(
                'random_forest',
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=self.parallel_jobs,
                    class_weight='balanced'
                ),
                num_input_features=X_train.shape[1]
            )),
            ('xgb', self._build_pipeline(
                'xgboost',
                xgb.XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    n_jobs=self.parallel_jobs
                ),
                num_input_features=X_train.shape[1]
            ))
        ]
        
        # Stacking with logistic regression as final estimator
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=cv,
            n_jobs=self.parallel_jobs,
            passthrough=False
        )
        
        stacking_clf.fit(X_train, y_train)
        self.models['stacking_ensemble'] = stacking_clf
        
        logger.info("Stacking ensemble trained successfully")
        return stacking_clf
    
    def train_voting_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> VotingClassifier:
        """
        Create a soft voting ensemble of all base models.
        """
        logger.info("Training voting ensemble...")
        
        estimators = [
            ('lr', self._build_pipeline(
                'logistic_regression',
                LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
                num_input_features=X_train.shape[1]
            )),
            ('rf', self._build_pipeline(
                'random_forest',
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=self.parallel_jobs,
                    class_weight='balanced'
                ),
                num_input_features=X_train.shape[1]
            )),
            ('xgb', self._build_pipeline(
                'xgboost',
                xgb.XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    n_jobs=self.parallel_jobs
                ),
                num_input_features=X_train.shape[1]
            ))
        ]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=self.parallel_jobs
        )
        
        voting_clf.fit(X_train, y_train)
        self.models['voting_ensemble'] = voting_clf
        
        logger.info("Voting ensemble trained successfully")
        return voting_clf


def train_all_models(X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_train: pd.Series,
                    y_test: pd.Series,
                    antibiotic: str,
                    cv: int = 5,
                    results_dir: str = "results",
                    figures_dir: Optional[str] = None,
                    metrics_dir: Optional[str] = None,
                    use_smote: bool = True,
                    use_feature_selection: bool = True,
                    n_features: int = 30,
                    use_ensemble: bool = True) -> ModelTrainer:
    """
    Train all models with improvements: SMOTE, feature selection, threshold tuning, ensembles.
    
    Args:
        X_train, X_test: Feature dataframes
        y_train, y_test: Target series
        antibiotic: Name of the antibiotic being predicted
        cv: Cross-validation folds
        use_smote: Apply SMOTE for class balancing
        use_feature_selection: Use RFE for feature selection
        n_features: Number of features to select if using feature selection
        use_ensemble: Train ensemble models (stacking + voting)
    """
    trainer = ModelTrainer(
        results_dir=results_dir,
        figures_dir=figures_dir,
        metrics_dir=metrics_dir,
        use_smote=use_smote,
        use_feature_selection=use_feature_selection,
        n_features=n_features,
        artifact_prefix=antibiotic
    )
    feature_names = X_train.columns.tolist()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING MODELS FOR {antibiotic}")
    logger.info(f"{'='*60}\n")
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    logger.info(f"Number of features: {X_train.shape[1]}")
    logger.info(f"Class distribution (train): {y_train.value_counts().to_dict()}")
    
    X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
    y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
    selected_features = feature_names
    
    # Train base models
    models_to_train = {
        'logistic_regression': trainer.train_logistic_regression,
        'random_forest': trainer.train_random_forest,
        'xgboost': trainer.train_xgboost
    }
    
    optimal_thresholds = {}
    
    for model_name, train_func in models_to_train.items():
        try:
            # Train
            model = train_func(X_train_array, y_train_array, cv=cv)
            
            # Find optimal threshold from training-only out-of-fold predictions
            opt_threshold = trainer.find_optimal_threshold(model, X_train_array, y_train_array, cv=cv)
            optimal_thresholds[model_name] = opt_threshold
            
            # Evaluate with default threshold
            trainer.evaluate_model(model, X_test_array, y_test, model_name)
            
            # Also evaluate with optimal threshold
            y_proba = model.predict_proba(X_test_array)[:, 1]
            y_pred_optimal = (y_proba >= opt_threshold).astype(int)
            
            logger.info(f"\n{model_name} with optimal threshold ({opt_threshold:.2f}):")
            logger.info(f"  Accuracy: {accuracy_score(y_test_array, y_pred_optimal):.4f}")
            logger.info(f"  Precision: {precision_score(y_test_array, y_pred_optimal, zero_division=0):.4f}")
            logger.info(f"  Recall: {recall_score(y_test_array, y_pred_optimal, zero_division=0):.4f}")
            logger.info(f"  F1: {f1_score(y_test_array, y_pred_optimal, zero_division=0):.4f}")
            
            # Plot confusion matrix
            y_pred = model.predict(X_test_array)
            trainer.plot_confusion_matrix(y_test, y_pred, model_name)
            
            # Save model
            trainer.save_model(model, model_name, antibiotic)

            _, selected_features = trainer._unwrap_estimator_and_features(model, feature_names)
            logger.info(f"Active features for {model_name}: {selected_features[:10]}...")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Train ensemble models
    if use_ensemble:
        try:
            logger.info("\n" + "="*40)
            logger.info("TRAINING ENSEMBLE MODELS")
            logger.info("="*40 + "\n")
            
            # Stacking ensemble
            stacking_model = trainer.train_stacking_ensemble(X_train_array, y_train_array, cv=cv)
            trainer.evaluate_model(stacking_model, X_test_array, y_test, 'stacking_ensemble')
            y_pred_stack = stacking_model.predict(X_test_array)
            trainer.plot_confusion_matrix(y_test, y_pred_stack, 'stacking_ensemble')
            trainer.save_model(stacking_model, 'stacking_ensemble', antibiotic)
            
            # Voting ensemble
            voting_model = trainer.train_voting_ensemble(X_train_array, y_train_array)
            trainer.evaluate_model(voting_model, X_test_array, y_test, 'voting_ensemble')
            y_pred_vote = voting_model.predict(X_test_array)
            trainer.plot_confusion_matrix(y_test, y_pred_vote, 'voting_ensemble')
            trainer.save_model(voting_model, 'voting_ensemble', antibiotic)
            
        except Exception as e:
            logger.error(f"Error training ensemble models: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare all models
    trainer.compare_models()
    
    # Plot ROC curves
    trainer.plot_roc_curves(X_test_array, y_test)
    
    # Plot feature importance for tree models (using selected features)
    if 'random_forest' in trainer.models:
        trainer.plot_feature_importance(
            trainer.models['random_forest'],
            selected_features,
            'random_forest'
        )
    
    if 'xgboost' in trainer.models:
        trainer.plot_feature_importance(
            trainer.models['xgboost'],
            selected_features,
            'xgboost'
        )
    
    # Save results
    trainer.save_results(antibiotic)
    
    # Log optimal thresholds
    logger.info("\n" + "="*40)
    logger.info("OPTIMAL THRESHOLDS (for deployment)")
    logger.info("="*40)
    for model_name, threshold in optimal_thresholds.items():
        logger.info(f"  {model_name}: {threshold:.2f}")
    
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
