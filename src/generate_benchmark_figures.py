"""
Generate comparison figures for strict profile-vs-gene benchmark results.
"""

from __future__ import annotations

import json
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
FIGURES_DIR = RESULTS_LAYOUT["result_figures"]
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_RESULTS = RESULTS_LAYOUT["current_metrics"] / "results_CIPROFLOXACIN_20260401_163441.json"
GENE_RESULTS = RESULTS_LAYOUT["benchmark_metrics"] / "results_CIPROFLOXACIN_GENES_EUCAST_20260401_172734.json"

PALETTE = {
    "blue": "#2A6F97",
    "orange": "#F4A261",
    "purple": "#7B6DCC",
    "slate": "#5C677D"
}


def load_results(path: Path, source_label: str) -> pd.DataFrame:
    with path.open() as f:
        data = json.load(f)

    rows = []
    for model_name, payload in data.items():
        metrics = payload["metrics"].copy()
        metrics["model"] = model_name
        metrics["feature_source"] = source_label
        rows.append(metrics)

    return pd.DataFrame(rows)


def get_sample_counts() -> pd.DataFrame:
    profile = prepare_training_data("CIPROFLOXACIN", feature_source="profiles", testing_standard="EUCAST")
    genes = prepare_training_data("CIPROFLOXACIN", feature_source="genes", testing_standard="EUCAST")

    rows = []
    for label, split in [("Profiles", profile), ("Genes (EUCAST)", genes)]:
        X_train, X_test, _, _ = split
        rows.extend([
            {"feature_source": label, "split": "Train", "count": len(X_train)},
            {"feature_source": label, "split": "Test", "count": len(X_test)},
            {"feature_source": label, "split": "Total", "count": len(X_train) + len(X_test)},
        ])

    return pd.DataFrame(rows)


def plot_profile_model_summary(profile_df: pd.DataFrame) -> Path:
    ordered = profile_df.sort_values("roc_auc", ascending=False).copy()
    display_names = [name.replace("_", " ").title() for name in ordered["model"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(ordered))
    width = 0.35

    ax.bar([i - width / 2 for i in x], ordered["roc_auc"], width=width, label="ROC-AUC", color=PALETTE["blue"])
    ax.bar([i + width / 2 for i in x], ordered["f1_score"], width=width, label="F1", color=PALETTE["orange"])

    ax.set_xticks(list(x))
    ax.set_xticklabels(display_names, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Ciprofloxacin Strict Profile Models")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    output = FIGURES_DIR / "ciprofloxacin_strict_profile_models.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    return output


def plot_source_comparison(profile_df: pd.DataFrame, gene_df: pd.DataFrame) -> Path:
    compare_df = pd.concat([profile_df, gene_df], ignore_index=True)
    compare_df = compare_df[compare_df["model"].isin(["logistic_regression", "random_forest", "xgboost"])].copy()

    model_order = ["logistic_regression", "random_forest", "xgboost"]
    source_order = ["Profiles", "Genes (EUCAST)"]
    auc_lookup = {
        (row["model"], row["feature_source"]): row["roc_auc"]
        for _, row in compare_df.iterrows()
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    x = range(len(model_order))
    width = 0.34

    ax.bar(
        [i - width / 2 for i in x],
        [auc_lookup.get((model, source_order[0]), 0) for model in model_order],
        width=width,
        label=source_order[0],
        color=PALETTE["blue"]
    )
    ax.bar(
        [i + width / 2 for i in x],
        [auc_lookup.get((model, source_order[1]), 0) for model in model_order],
        width=width,
        label=source_order[1],
        color=PALETTE["orange"]
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels([name.replace("_", " ").title() for name in model_order])
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Ciprofloxacin Feature Source Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    output = FIGURES_DIR / "ciprofloxacin_profiles_vs_genes_auc.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    return output


def plot_sample_coverage(sample_df: pd.DataFrame) -> Path:
    split_order = ["Train", "Test", "Total"]
    source_order = ["Profiles", "Genes (EUCAST)"]
    count_lookup = {
        (row["feature_source"], row["split"]): row["count"]
        for _, row in sample_df.iterrows()
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(len(split_order))
    width = 0.34

    ax.bar(
        [i - width / 2 for i in x],
        [count_lookup.get((source_order[0], split), 0) for split in split_order],
        width=width,
        label=source_order[0],
        color=PALETTE["purple"]
    )
    ax.bar(
        [i + width / 2 for i in x],
        [count_lookup.get((source_order[1], split), 0) for split in split_order],
        width=width,
        label=source_order[1],
        color=PALETTE["orange"]
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(split_order)
    ax.set_ylabel("Number of Isolates")
    ax.set_title("Ciprofloxacin Sample Coverage by Feature Source")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    output = FIGURES_DIR / "ciprofloxacin_sample_coverage.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    return output


def main():
    profile_df = load_results(PROFILE_RESULTS, "Profiles")
    gene_df = load_results(GENE_RESULTS, "Genes (EUCAST)")
    sample_df = get_sample_counts()

    outputs = [
        plot_profile_model_summary(profile_df),
        plot_source_comparison(profile_df, gene_df),
        plot_sample_coverage(sample_df),
    ]

    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
