"""
Utility functions for E. coli Antibiotic Resistance Project
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[str] = None, level: str = "INFO"):
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (if None, logs to console only)
        level: Logging level
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: str = "config.py") -> Dict[str, Any]:
    """
    Load configuration from config file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary of configuration values
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    config_dict = {k: v for k, v in vars(config).items() 
                   if not k.startswith('_')}
    
    return config_dict


def save_json(data: Dict, filepath: str):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        return obj
    
    serializable_data = json.loads(
        json.dumps(data, default=convert)
    )
    
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=4)
    
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Load JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary from JSON
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded JSON from {filepath}")
    return data


def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def check_class_balance(y: pd.Series, threshold: float = 0.1) -> Dict[str, Any]:
    """
    Check if classes are balanced and return statistics.
    
    Args:
        y: Target variable
        threshold: Threshold for considering classes imbalanced
        
    Returns:
        Dictionary with balance statistics
    """
    counts = y.value_counts()
    proportions = y.value_counts(normalize=True)
    
    min_proportion = proportions.min()
    max_proportion = proportions.max()
    imbalance_ratio = min_proportion / max_proportion if max_proportion > 0 else 0
    
    is_imbalanced = imbalance_ratio < threshold
    
    balance_info = {
        'counts': counts.to_dict(),
        'proportions': proportions.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'is_imbalanced': is_imbalanced
    }
    
    logger.info(f"Class balance check:")
    logger.info(f"  Counts: {counts.to_dict()}")
    logger.info(f"  Proportions: {proportions.to_dict()}")
    logger.info(f"  Imbalance ratio: {imbalance_ratio:.4f}")
    logger.info(f"  Imbalanced: {is_imbalanced}")
    
    return balance_info


def summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive dataset summary.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'shape': df.shape,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        summary['categorical_counts'] = {
            col: df[col].value_counts().to_dict()
            for col in categorical_cols
        }
    
    return summary


def filter_low_variance_features(X: pd.DataFrame, 
                                 threshold: float = 0.01) -> pd.DataFrame:
    """
    Remove features with low variance.
    
    Args:
        X: Feature matrix
        threshold: Variance threshold
        
    Returns:
        Filtered feature matrix
    """
    from sklearn.feature_selection import VarianceThreshold
    
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    
    selected_features = X.columns[selector.get_support()].tolist()
    removed_features = X.columns[~selector.get_support()].tolist()
    
    logger.info(f"Variance threshold filtering:")
    logger.info(f"  Original features: {X.shape[1]}")
    logger.info(f"  Selected features: {len(selected_features)}")
    logger.info(f"  Removed features: {len(removed_features)}")
    
    return pd.DataFrame(X_filtered, columns=selected_features, index=X.index)


def get_top_features(model: Any, 
                    feature_names: List[str],
                    n: int = 20) -> pd.DataFrame:
    """
    Get top N most important features from a trained model.
    
    Args:
        model: Trained model with feature_importances_ or coef_
        feature_names: List of feature names
        n: Number of top features to return
        
    Returns:
        DataFrame with top features and their importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Model does not have feature importance attributes")
        return pd.DataFrame()
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feature_importance.head(n)


def create_result_summary(results: Dict[str, Any], 
                         output_file: str = "results/summary.txt"):
    """
    Create human-readable summary of results.
    
    Args:
        results: Dictionary of results
        output_file: Path to output text file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("E. COLI ANTIBIOTIC RESISTANCE PREDICTION - RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for key, value in results.items():
            f.write(f"\n{key.upper()}\n")
            f.write("-" * 80 + "\n")
            
            if isinstance(value, dict):
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"  {value}\n")
    
    logger.info(f"Result summary saved to {output_path}")


def validate_data_files(file_paths: List[str]) -> bool:
    """
    Check if required data files exist.
    
    Args:
        file_paths: List of file paths to check
        
    Returns:
        True if all files exist, False otherwise
    """
    all_exist = True
    
    for filepath in file_paths:
        path = Path(filepath)
        if not path.exists():
            logger.error(f"Required file not found: {filepath}")
            all_exist = False
        else:
            logger.info(f"Found: {filepath}")
    
    return all_exist


def merge_dicts(*dicts: Dict) -> Dict:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Variable number of dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def format_metric_table(metrics_dict: Dict[str, Dict[str, float]]) -> str:
    """
    Format metrics dictionary as a pretty table string.
    
    Args:
        metrics_dict: Dictionary of model names to metrics
        
    Returns:
        Formatted table string
    """
    df = pd.DataFrame(metrics_dict).T
    return df.to_string(float_format=lambda x: f"{x:.4f}")


def safe_divide(numerator: float, denominator: float, 
                default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is 0.
    """
    return numerator / denominator if denominator != 0 else default
