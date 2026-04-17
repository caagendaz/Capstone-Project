"""
Generate compact comparison artifacts from saved benchmark run outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from main import prepare_training_data


RESULTS_LAYOUT = config.get_results_layout("BVBRC")
RESULTS_DIR = RESULTS_LAYOUT["root"]
BENCHMARK_METRICS_DIR = RESULTS_LAYOUT["benchmark_metrics"]
FIGURES_DIR = RESULTS_LAYOUT["result_figures"]
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "blue": "#2A6F97",
    "orange": "#F4A261",
    "purple": "#7B6DCC",
    "slate": "#5C677D"
}

PROFILES_RE = re.compile(r"^results_(?P<antibiotic>.+)_PROFILES_(?P<timestamp>\d{8}_\d{6})\.json$")
GENES_RE = re.compile(r"^results_(?P<antibiotic>.+)_GENES_(?P<standard>[A-Z]+)_(?P<timestamp>\d{8}_\d{6})\.json$")


def sanitize_name(value: str) -> str:
    """Match the artifact name sanitization used during training."""
    return "".join(
        char if char.isalnum() or char in ("_", "-") else "_"
        for char in value
    ).strip("_")


def antibiotic_name_map() -> dict[str, str]:
    """Map sanitized artifact-safe antibiotic names back to the original labels."""
    candidate_names = set()
    for csv_path in [
        Path("data/processed/cleaned_resistance_data.csv"),
        Path("data/processed/merged_data_EUCAST.csv"),
        Path("data/processed/merged_data_CLSI.csv"),
    ]:
        if csv_path.exists():
            df = pd.read_csv(csv_path, usecols=["antibiotic"])
            candidate_names.update(df["antibiotic"].dropna().astype(str).unique())

    return {sanitize_name(name): name for name in candidate_names}


def latest_result_files(results_dir: Path) -> list[tuple[Path, dict[str, str]]]:
    """Select the latest JSON file for each antibiotic/source/standard slice."""
    latest: dict[tuple[str, str, str], tuple[str, Path, dict[str, str]]] = {}

    for path in results_dir.glob("results_*.json"):
        match = PROFILES_RE.match(path.name)
        if match:
            meta = {
                "antibiotic_key": match.group("antibiotic"),
                "feature_source": "profiles",
                "testing_standard": "ALL",
                "timestamp": match.group("timestamp")
            }
        else:
            match = GENES_RE.match(path.name)
            if not match:
                continue
            meta = {
                "antibiotic_key": match.group("antibiotic"),
                "feature_source": "genes",
                "testing_standard": match.group("standard"),
                "timestamp": match.group("timestamp")
            }

        key = (meta["antibiotic_key"], meta["feature_source"], meta["testing_standard"])
        current = latest.get(key)
        if current is None or meta["timestamp"] > current[0]:
            latest[key] = (meta["timestamp"], path, meta)

    selected = [(payload[1], payload[2]) for payload in latest.values()]
    selected.sort(key=lambda item: (item[1]["antibiotic_key"], item[1]["feature_source"], item[1]["testing_standard"]))
    return selected


def load_results_rows(path: Path,
                      meta: dict[str, str],
                      name_lookup: dict[str, str]) -> list[dict]:
    """Expand one result JSON into per-model summary rows with sample counts."""
    antibiotic = name_lookup.get(meta["antibiotic_key"], meta["antibiotic_key"].replace("_", " "))
    X_train, X_test, _, _ = prepare_training_data(
        antibiotic=antibiotic,
        feature_source=meta["feature_source"],
        testing_standard=meta["testing_standard"] if meta["feature_source"] == "genes" else "EUCAST"
    )

    with path.open() as f:
        data = json.load(f)

    rows = []
    for model_name, payload in data.items():
        metrics = payload["metrics"]
        rows.append({
            "antibiotic": antibiotic,
            "feature_source": meta["feature_source"],
            "testing_standard": meta["testing_standard"],
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_total": len(X_train) + len(X_test),
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "result_file": path.name
        })

    return rows


def build_compiled_summary(results_dir: Path) -> pd.DataFrame:
    """Build a combined benchmark summary from saved result JSONs."""
    name_lookup = antibiotic_name_map()
    rows: list[dict] = []

    for path, meta in latest_result_files(results_dir):
        rows.extend(load_results_rows(path, meta, name_lookup))

    if not rows:
        raise FileNotFoundError("No benchmark result JSON files were found to summarize.")

    summary_df = pd.DataFrame(rows).sort_values(
        ["antibiotic", "feature_source", "testing_standard", "roc_auc"],
        ascending=[True, True, True, False]
    )
    return summary_df


def build_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row-per-antibiotic comparison table using best model per source."""
    best_df = (
        summary_df.sort_values("roc_auc", ascending=False)
        .groupby(["antibiotic", "feature_source", "testing_standard"], as_index=False)
        .first()
    )

    profiles = (
        best_df[best_df["feature_source"] == "profiles"]
        .rename(columns={
            "model": "profiles_best_model",
            "roc_auc": "profiles_roc_auc",
            "f1_score": "profiles_f1_score",
            "n_total": "profiles_n_total"
        })[
            ["antibiotic", "profiles_best_model", "profiles_roc_auc", "profiles_f1_score", "profiles_n_total"]
        ]
    )

    genes = (
        best_df[(best_df["feature_source"] == "genes") & (best_df["testing_standard"] == "EUCAST")]
        .rename(columns={
            "model": "genes_best_model",
            "roc_auc": "genes_roc_auc",
            "f1_score": "genes_f1_score",
            "n_total": "genes_n_total"
        })[
            ["antibiotic", "genes_best_model", "genes_roc_auc", "genes_f1_score", "genes_n_total"]
        ]
    )

    comparison = profiles.merge(genes, on="antibiotic", how="inner")
    comparison["auc_delta_profiles_minus_genes"] = (
        comparison["profiles_roc_auc"] - comparison["genes_roc_auc"]
    )
    comparison = comparison.sort_values("profiles_roc_auc", ascending=False)
    return comparison


def plot_best_auc_comparison(comparison_df: pd.DataFrame, output_path: Path) -> Path:
    """Plot best ROC-AUC by antibiotic for profiles vs genes."""
    display_df = comparison_df.copy()
    display_df["label"] = display_df["antibiotic"].str.replace("/", "/\n", regex=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(display_df))
    width = 0.36

    ax.bar(
        [i - width / 2 for i in x],
        display_df["profiles_roc_auc"],
        width=width,
        color=PALETTE["blue"],
        label="Profiles"
    )
    ax.bar(
        [i + width / 2 for i in x],
        display_df["genes_roc_auc"],
        width=width,
        color=PALETTE["orange"],
        label="Genes (EUCAST)"
    )

    for i, row in enumerate(display_df.itertuples(index=False)):
        ax.text(i - width / 2, row.profiles_roc_auc + 0.015, row.profiles_best_model.replace("_", " "),
                ha="center", va="bottom", rotation=90, fontsize=8, color=PALETTE["slate"])
        ax.text(i + width / 2, row.genes_roc_auc + 0.015, row.genes_best_model.replace("_", " "),
                ha="center", va="bottom", rotation=90, fontsize=8, color=PALETTE["slate"])

    ax.set_xticks(list(x))
    ax.set_xticklabels(display_df["label"])
    ax.set_ylabel("Best ROC-AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Best ROC-AUC by Antibiotic and Feature Source")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate summary artifacts from benchmark results.")
    parser.add_argument(
        "--results-dir",
        default=str(BENCHMARK_METRICS_DIR),
        help="Directory containing benchmark result JSON files."
    )
    parser.add_argument(
        "--label",
        default="latest",
        help="Short suffix to use in the generated output filenames."
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary_df = build_compiled_summary(results_dir)
    comparison_df = build_comparison_table(summary_df)

    compiled_summary_path = BENCHMARK_METRICS_DIR / f"benchmark_summary_compiled_{args.label}.csv"
    comparison_path = BENCHMARK_METRICS_DIR / f"benchmark_feature_source_comparison_{args.label}.csv"
    figure_path = FIGURES_DIR / f"benchmark_best_auc_by_source_{args.label}.png"

    summary_df.to_csv(compiled_summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    plot_best_auc_comparison(comparison_df, figure_path)

    print(f"Saved {compiled_summary_path}")
    print(f"Saved {comparison_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
