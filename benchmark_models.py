"""
Batch benchmark runner for comparing phenotype-profile and AMR-gene models
across multiple antibiotics.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

import config
from main import run_model_training_with_source, run_model_validation
from src.utils import setup_logging

setup_logging(log_file="results/benchmark.log", level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


def load_antibiotic_counts(feature_source: str, testing_standard: str) -> pd.Series:
    """Load antibiotic counts for one feature source."""
    if feature_source == "profiles":
        df = pd.read_csv(config.PROCESSED_DATA_DIR / "cleaned_resistance_data.csv")
    else:
        df = pd.read_csv(config.PROCESSED_DATA_DIR / f"merged_data_{testing_standard}.csv")

    return df["antibiotic"].value_counts()


def resolve_antibiotics(feature_source: str,
                        testing_standard: str,
                        requested_antibiotics: List[str] | None,
                        top_n: int,
                        min_samples: int) -> List[str]:
    """Resolve which antibiotics to benchmark for a given dataset slice."""
    counts = load_antibiotic_counts(feature_source, testing_standard)
    eligible = counts[counts >= min_samples]

    if requested_antibiotics:
        requested = [antibiotic.upper() for antibiotic in requested_antibiotics]
        selected = [antibiotic for antibiotic in requested if antibiotic in eligible.index]
    else:
        selected = eligible.head(top_n).index.tolist()

    logger.info(
        "Selected antibiotics for %s/%s: %s",
        feature_source,
        testing_standard,
        selected
    )
    return selected


def build_summary_rows(trainer,
                       antibiotic: str,
                       feature_source: str,
                       testing_standard: str,
                       X_train,
                       X_test) -> List[dict]:
    """Flatten model metrics into benchmark summary rows."""
    rows = []
    for model_name, result in trainer.results.items():
        metrics = result["metrics"]
        rows.append({
            "antibiotic": antibiotic,
            "feature_source": feature_source,
            "testing_standard": testing_standard if feature_source == "genes" else "ALL",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_total": len(X_train) + len(X_test),
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"]
        })
    return rows


def run_benchmarks(feature_sources: List[str],
                   testing_standards: List[str],
                   antibiotics: List[str] | None,
                   top_n: int,
                   min_samples: int,
                   cv: int,
                   use_ensemble: bool,
                   run_validation: bool) -> Path:
    """Run benchmark combinations and save summary CSVs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows: List[dict] = []

    for feature_source in feature_sources:
        standards = testing_standards if feature_source == "genes" else ["ALL"]

        for standard in standards:
            selected_antibiotics = resolve_antibiotics(
                feature_source=feature_source,
                testing_standard=standard if standard != "ALL" else "EUCAST",
                requested_antibiotics=antibiotics,
                top_n=top_n,
                min_samples=min_samples
            )

            for antibiotic in selected_antibiotics:
                logger.info(
                    "Benchmarking %s with %s features (%s)",
                    antibiotic,
                    feature_source,
                    standard
                )

                trainer, X_train, X_test, y_train, y_test = run_model_training_with_source(
                    antibiotic=antibiotic,
                    feature_source=feature_source,
                    testing_standard=standard if standard != "ALL" else "EUCAST",
                    cv=cv,
                    use_ensemble=use_ensemble,
                    metrics_dir=str(config.BENCHMARK_METRICS_DIR)
                )

                if trainer is None:
                    logger.warning("Skipping %s because training data could not be prepared", antibiotic)
                    continue

                summary_rows.extend(
                    build_summary_rows(
                        trainer=trainer,
                        antibiotic=antibiotic,
                        feature_source=feature_source,
                        testing_standard=standard,
                        X_train=X_train,
                        X_test=X_test
                    )
                )

                if run_validation:
                    run_model_validation(trainer, X_train, X_test, y_train, y_test)

    if not summary_rows:
        raise RuntimeError("No benchmark results were generated.")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["antibiotic", "feature_source", "testing_standard", "roc_auc"],
        ascending=[True, True, True, False]
    )
    summary_path = config.BENCHMARK_METRICS_DIR / f"benchmark_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)

    best_df = (
        summary_df.sort_values("roc_auc", ascending=False)
        .groupby(["antibiotic", "feature_source", "testing_standard"], as_index=False)
        .first()
    )
    best_path = config.BENCHMARK_METRICS_DIR / f"benchmark_best_models_{timestamp}.csv"
    best_df.to_csv(best_path, index=False)

    logger.info("Saved benchmark summary to %s", summary_path)
    logger.info("Saved benchmark best-model summary to %s", best_path)
    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-antibiotic benchmarks across feature sources."
    )
    parser.add_argument(
        "--feature-sources",
        nargs="+",
        choices=["profiles", "genes"],
        default=["profiles", "genes"],
        help="Feature sources to benchmark"
    )
    parser.add_argument(
        "--testing-standards",
        nargs="+",
        choices=["CLSI", "EUCAST"],
        default=["EUCAST", "CLSI"],
        help="Testing standards to use for gene-based benchmarks"
    )
    parser.add_argument(
        "--antibiotics",
        nargs="+",
        help="Optional explicit list of antibiotics to benchmark"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top antibiotics to benchmark when --antibiotics is not supplied"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=200,
        help="Minimum rows required for an antibiotic to be benchmarked"
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Cross-validation folds for benchmark training"
    )
    parser.add_argument(
        "--use-ensemble",
        action="store_true",
        help="Also train ensemble models during benchmarking"
    )
    parser.add_argument(
        "--run-validation",
        action="store_true",
        help="Run the expensive validation suite for each benchmarked model"
    )

    args = parser.parse_args()

    run_benchmarks(
        feature_sources=args.feature_sources,
        testing_standards=args.testing_standards,
        antibiotics=args.antibiotics,
        top_n=args.top_n,
        min_samples=args.min_samples,
        cv=args.cv,
        use_ensemble=args.use_ensemble,
        run_validation=args.run_validation
    )


if __name__ == "__main__":
    main()
