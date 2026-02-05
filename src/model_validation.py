"""
Model Validation Module for E. coli Antibiotic Resistance Project

Performs:
1. Permutation tests (to ensure accuracy > random)
2. Monte Carlo simulations (for sampling variation and stability)
3. Confidence interval estimation
4. Statistical significance testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
import logging
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Validates model performance through statistical testing.
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def permutation_test(self,
                        model: Any,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        n_permutations: int = 1000,
                        metric: str = 'accuracy',
                        random_state: int = 42) -> Dict[str, float]:
        """
        Perform permutation test to assess if model performance is better than random.
        
        Null hypothesis: Model predictions are independent of true labels.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: True test labels
            n_permutations: Number of permutations
            metric: Metric to evaluate ('accuracy', 'f1', 'roc_auc')
            
        Returns:
            Dictionary with test results including p-value
        """
        logger.info(f"\nPerforming permutation test with {n_permutations} permutations...")
        logger.info(f"Metric: {metric}")
        
        # Get actual model performance
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        if metric == 'accuracy':
            actual_score = accuracy_score(y_test, y_pred)
        elif metric == 'f1':
            actual_score = f1_score(y_test, y_pred, zero_division=0)
        elif metric == 'roc_auc' and y_pred_proba is not None:
            actual_score = roc_auc_score(y_test, y_pred_proba)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        logger.info(f"Actual {metric}: {actual_score:.4f}")
        
        # Permutation test
        np.random.seed(random_state)
        permuted_scores = []
        
        for i in tqdm(range(n_permutations), desc="Permutation test"):
            # Randomly permute labels
            y_permuted = np.random.permutation(y_test)
            
            # Calculate metric with permuted labels
            if metric == 'accuracy':
                score = accuracy_score(y_permuted, y_pred)
            elif metric == 'f1':
                score = f1_score(y_permuted, y_pred, zero_division=0)
            elif metric == 'roc_auc':
                score = roc_auc_score(y_permuted, y_pred_proba)
            
            permuted_scores.append(score)
        
        permuted_scores = np.array(permuted_scores)
        
        # Calculate p-value (one-tailed test)
        p_value = (permuted_scores >= actual_score).sum() / n_permutations
        
        # Effect size (standardized difference)
        effect_size = (actual_score - permuted_scores.mean()) / permuted_scores.std()
        
        results = {
            'actual_score': actual_score,
            'permuted_mean': permuted_scores.mean(),
            'permuted_std': permuted_scores.std(),
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05
        }
        
        logger.info(f"\nPermutation Test Results:")
        logger.info(f"  Actual {metric}: {actual_score:.4f}")
        logger.info(f"  Permuted mean: {permuted_scores.mean():.4f} ± {permuted_scores.std():.4f}")
        logger.info(f"  P-value: {p_value:.4f}")
        logger.info(f"  Effect size: {effect_size:.4f}")
        logger.info(f"  Significant at α=0.05: {results['significant']}")
        
        return results, permuted_scores
    
    def plot_permutation_test(self,
                             actual_score: float,
                             permuted_scores: np.ndarray,
                             metric: str,
                             model_name: str):
        """
        Plot permutation test results.
        """
        plt.figure(figsize=(10, 6))
        
        # Histogram of permuted scores
        plt.hist(permuted_scores, bins=50, alpha=0.7, color='skyblue', 
                edgecolor='black', label='Permuted distribution')
        
        # Actual score
        plt.axvline(actual_score, color='red', linestyle='--', linewidth=2,
                   label=f'Actual score = {actual_score:.4f}')
        
        # Mean of permuted scores
        plt.axvline(permuted_scores.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Permuted mean = {permuted_scores.mean():.4f}')
        
        plt.xlabel(metric.replace('_', ' ').title())
        plt.ylabel('Frequency')
        plt.title(f'Permutation Test - {model_name}')
        plt.legend()
        plt.grid(alpha=0.3)
        
        output_path = self.results_dir / f"permutation_test_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved permutation test plot to {output_path}")
        plt.close()
    
    def monte_carlo_simulation(self,
                              model: Any,
                              X: np.ndarray,
                              y: np.ndarray,
                              n_iterations: int = 100,
                              test_size: float = 0.2,
                              metrics: Optional[List[str]] = None,
                              random_state: int = 42) -> Dict[str, np.ndarray]:
        """
        Perform Monte Carlo simulation to assess model stability.
        
        Repeatedly samples train/test splits and evaluates model performance
        to estimate variance due to sampling.
        
        Args:
            model: Model class or instance to train
            X: Feature matrix
            y: Labels
            n_iterations: Number of Monte Carlo iterations
            test_size: Fraction of data for test set
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric scores across iterations
        """
        logger.info(f"\nPerforming Monte Carlo simulation with {n_iterations} iterations...")
        logger.info(f"Test size: {test_size}")
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Store results
        mc_results = {metric: [] for metric in metrics}
        
        np.random.seed(random_state)
        
        for i in tqdm(range(n_iterations), desc="Monte Carlo simulation"):
            # Random train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                stratify=y,
                random_state=random_state + i
            )
            
            # Clone and train model
            try:
                from sklearn.base import clone
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                
                # Predict
                y_pred = model_clone.predict(X_test)
                y_pred_proba = model_clone.predict_proba(X_test)[:, 1] if hasattr(model_clone, 'predict_proba') else None
                
                # Calculate metrics
                if 'accuracy' in metrics:
                    mc_results['accuracy'].append(accuracy_score(y_test, y_pred))
                if 'precision' in metrics:
                    mc_results['precision'].append(precision_score(y_test, y_pred, zero_division=0))
                if 'recall' in metrics:
                    mc_results['recall'].append(recall_score(y_test, y_pred, zero_division=0))
                if 'f1' in metrics:
                    mc_results['f1'].append(f1_score(y_test, y_pred, zero_division=0))
                if 'roc_auc' in metrics and y_pred_proba is not None:
                    mc_results['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))
                    
            except Exception as e:
                logger.warning(f"Error in iteration {i}: {e}")
                continue
        
        # Convert to arrays and compute statistics
        for metric in metrics:
            scores = np.array(mc_results[metric])
            logger.info(f"\n{metric.upper()}:")
            logger.info(f"  Mean: {scores.mean():.4f}")
            logger.info(f"  Std: {scores.std():.4f}")
            logger.info(f"  Min: {scores.min():.4f}")
            logger.info(f"  Max: {scores.max():.4f}")
            logger.info(f"  Median: {np.median(scores):.4f}")
        
        return mc_results
    
    def plot_monte_carlo_results(self,
                                mc_results: Dict[str, np.ndarray],
                                model_name: str):
        """
        Plot Monte Carlo simulation results.
        """
        n_metrics = len(mc_results)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, (metric, scores) in zip(axes, mc_results.items()):
            scores_array = np.array(scores)
            
            # Histogram
            ax.hist(scores_array, bins=30, alpha=0.7, color='skyblue', 
                   edgecolor='black')
            
            # Mean and confidence interval
            mean = scores_array.mean()
            ci_lower, ci_upper = np.percentile(scores_array, [2.5, 97.5])
            
            ax.axvline(mean, color='red', linestyle='--', linewidth=2,
                      label=f'Mean = {mean:.4f}')
            ax.axvline(ci_lower, color='green', linestyle=':', linewidth=1.5)
            ax.axvline(ci_upper, color='green', linestyle=':', linewidth=1.5,
                      label=f'95% CI [{ci_lower:.4f}, {ci_upper:.4f}]')
            
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'{metric.upper()} Distribution')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'Monte Carlo Simulation - {model_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.results_dir / f"monte_carlo_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Monte Carlo plot to {output_path}")
        plt.close()
    
    def compute_confidence_intervals(self,
                                    scores: np.ndarray,
                                    confidence: float = 0.95,
                                    method: str = 'percentile') -> Tuple[float, float, float]:
        """
        Compute confidence intervals for performance metrics.
        
        Args:
            scores: Array of scores from multiple runs
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            method: 'percentile' or 'normal'
            
        Returns:
            mean, lower_bound, upper_bound
        """
        mean = scores.mean()
        
        if method == 'percentile':
            # Bootstrap percentile method
            alpha = 1 - confidence
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower = np.percentile(scores, lower_percentile)
            upper = np.percentile(scores, upper_percentile)
            
        elif method == 'normal':
            # Assume normal distribution
            std = scores.std()
            n = len(scores)
            se = std / np.sqrt(n)
            
            # t-distribution for small samples
            if n < 30:
                t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
            else:
                t_critical = stats.norm.ppf((1 + confidence) / 2)
            
            margin = t_critical * se
            lower = mean - margin
            upper = mean + margin
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return mean, lower, upper
    
    def bootstrap_confidence_interval(self,
                                     model: Any,
                                     X_test: np.ndarray,
                                     y_test: np.ndarray,
                                     metric: str = 'accuracy',
                                     n_bootstrap: int = 1000,
                                     confidence: float = 0.95,
                                     random_state: int = 42) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for a specific metric.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: True test labels
            metric: Metric to compute CI for
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            mean, lower_bound, upper_bound
        """
        logger.info(f"\nComputing bootstrap CI for {metric}...")
        logger.info(f"Bootstrap samples: {n_bootstrap}")
        logger.info(f"Confidence level: {confidence}")
        
        np.random.seed(random_state)
        n_samples = len(y_test)
        bootstrap_scores = []
        
        # Get predictions once
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        for i in tqdm(range(n_bootstrap), desc="Bootstrap"):
            # Bootstrap sample indices
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Calculate metric on bootstrap sample
            if metric == 'accuracy':
                score = accuracy_score(y_test[indices], y_pred[indices])
            elif metric == 'precision':
                score = precision_score(y_test[indices], y_pred[indices], zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_test[indices], y_pred[indices], zero_division=0)
            elif metric == 'f1':
                score = f1_score(y_test[indices], y_pred[indices], zero_division=0)
            elif metric == 'roc_auc' and y_pred_proba is not None:
                # Check if both classes present in bootstrap sample
                if len(np.unique(y_test[indices])) > 1:
                    score = roc_auc_score(y_test[indices], y_pred_proba[indices])
                else:
                    continue
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            bootstrap_scores.append(score)
        
        bootstrap_scores = np.array(bootstrap_scores)
        mean, lower, upper = self.compute_confidence_intervals(
            bootstrap_scores, confidence, method='percentile'
        )
        
        logger.info(f"\n{metric.upper()} Bootstrap CI ({int(confidence*100)}%):")
        logger.info(f"  Mean: {mean:.4f}")
        logger.info(f"  Lower: {lower:.4f}")
        logger.info(f"  Upper: {upper:.4f}")
        logger.info(f"  Width: {upper - lower:.4f}")
        
        return mean, lower, upper
    
    def plot_confidence_intervals(self,
                                 ci_results: Dict[str, Tuple[float, float, float]],
                                 model_name: str):
        """
        Plot confidence intervals for multiple metrics.
        """
        metrics = list(ci_results.keys())
        means = [ci_results[m][0] for m in metrics]
        lowers = [ci_results[m][1] for m in metrics]
        uppers = [ci_results[m][2] for m in metrics]
        
        errors_lower = [m - l for m, l in zip(means, lowers)]
        errors_upper = [u - m for m, u in zip(means, uppers)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(metrics))
        ax.errorbar(x_pos, means, 
                   yerr=[errors_lower, errors_upper],
                   fmt='o', markersize=10, capsize=5, capthick=2,
                   color='steelblue', ecolor='darkblue')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylabel('Score')
        ax.set_title(f'Performance Metrics with 95% Confidence Intervals - {model_name}',
                    fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        output_path = self.results_dir / f"confidence_intervals_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confidence intervals plot to {output_path}")
        plt.close()
    
    def comprehensive_validation(self,
                                model: Any,
                                X_train: np.ndarray,
                                X_test: np.ndarray,
                                y_train: np.ndarray,
                                y_test: np.ndarray,
                                model_name: str,
                                n_permutations: int = 1000,
                                n_mc_iterations: int = 100,
                                n_bootstrap: int = 1000) -> Dict:
        """
        Run comprehensive validation including all tests.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPREHENSIVE VALIDATION - {model_name}")
        logger.info(f"{'='*60}\n")
        
        results = {}
        
        # 1. Permutation test
        perm_results, perm_scores = self.permutation_test(
            model, X_test, y_test,
            n_permutations=n_permutations,
            metric='accuracy'
        )
        results['permutation_test'] = perm_results
        self.plot_permutation_test(
            perm_results['actual_score'],
            perm_scores,
            'accuracy',
            model_name
        )
        
        # 2. Monte Carlo simulation
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])
        
        mc_results = self.monte_carlo_simulation(
            model, X_all, y_all,
            n_iterations=n_mc_iterations,
            test_size=0.2
        )
        results['monte_carlo'] = mc_results
        self.plot_monte_carlo_results(mc_results, model_name)
        
        # 3. Bootstrap confidence intervals
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        ci_results = {}
        
        for metric in metrics:
            try:
                mean, lower, upper = self.bootstrap_confidence_interval(
                    model, X_test, y_test,
                    metric=metric,
                    n_bootstrap=n_bootstrap
                )
                ci_results[metric] = (mean, lower, upper)
            except Exception as e:
                logger.warning(f"Could not compute CI for {metric}: {e}")
        
        results['confidence_intervals'] = ci_results
        self.plot_confidence_intervals(ci_results, model_name)
        
        logger.info(f"\nComprehensive validation complete for {model_name}")
        
        return results


def main():
    """
    Main validation pipeline.
    """
    # Example usage
    logger.info("Model validation pipeline ready")
    
    # Load trained model and data
    # model = joblib.load("models/xgboost_CIPROFLOXACIN_20260205_120000.pkl")
    # X_train, X_test, y_train, y_test = ...
    
    # validator = ModelValidator()
    # results = validator.comprehensive_validation(
    #     model, X_train, X_test, y_train, y_test,
    #     model_name='xgboost',
    #     n_permutations=1000,
    #     n_mc_iterations=100,
    #     n_bootstrap=1000
    # )


if __name__ == "__main__":
    main()
