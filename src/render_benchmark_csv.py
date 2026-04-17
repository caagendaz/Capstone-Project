"""
Render benchmark CSV files into chart and table images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

RESULTS_LAYOUT = config.get_results_layout("BVBRC")
RESULTS_DIR = RESULTS_LAYOUT["root"]
BENCHMARK_METRICS_DIR = RESULTS_LAYOUT["benchmark_metrics"]
FIGURES_DIR = RESULTS_LAYOUT["result_figures"]
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "blue": "#2A6F97",
    "orange": "#F4A261",
    "purple": "#7B6DCC",
    "slate": "#5C677D",
    "teal": "#2A9D8F"
}


def resolve_csv_path(explicit_path: str | None) -> Path:
    """Resolve the benchmark CSV to render."""
    if explicit_path:
        return Path(explicit_path)

    preferred = sorted(BENCHMARK_METRICS_DIR.glob("benchmark_feature_source_comparison_*.csv"))
    if preferred:
        return preferred[-1]

    fallback = sorted(BENCHMARK_METRICS_DIR.glob("benchmark_summary_compiled_*.csv"))
    if fallback:
        return fallback[-1]

    raise FileNotFoundError("No benchmark comparison or compiled summary CSV found in results/metrics/benchmark")


def build_best_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize either CSV format into one row per antibiotic comparison."""
    if {"profiles_roc_auc", "genes_roc_auc"}.issubset(df.columns):
        return df.copy()

    if {"antibiotic", "feature_source", "testing_standard", "model", "roc_auc", "f1_score", "n_total"}.issubset(df.columns):
        best_df = (
            df.sort_values("roc_auc", ascending=False)
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
        return comparison

    raise ValueError("Unsupported benchmark CSV format.")


def save_chart(comparison_df: pd.DataFrame, output_path: Path) -> Path:
    """Save grouped ROC-AUC chart."""
    display_df = comparison_df.sort_values("profiles_roc_auc", ascending=False).copy()
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

    ax.set_xticks(list(x))
    ax.set_xticklabels(display_df["label"])
    ax.set_ylabel("Best ROC-AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Benchmark Comparison by Antibiotic")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def save_table(comparison_df: pd.DataFrame, output_path: Path) -> Path:
    """Save formatted benchmark comparison table as an image."""
    table_df = comparison_df.copy()
    table_df = table_df.rename(columns={
        "antibiotic": "Antibiotic",
        "profiles_best_model": "Profiles Model",
        "profiles_roc_auc": "Profiles AUC",
        "profiles_f1_score": "Profiles F1",
        "profiles_n_total": "Profiles n",
        "genes_best_model": "Genes Model",
        "genes_roc_auc": "Genes AUC",
        "genes_f1_score": "Genes F1",
        "genes_n_total": "Genes n",
        "auc_delta_profiles_minus_genes": "AUC Delta"
    })

    for col in ["Profiles AUC", "Profiles F1", "Genes AUC", "Genes F1", "AUC Delta"]:
        table_df[col] = table_df[col].map(lambda value: f"{value:.3f}")

    table_df["Profiles Model"] = table_df["Profiles Model"].str.replace("_", " ").str.title()
    table_df["Genes Model"] = table_df["Genes Model"].str.replace("_", " ").str.title()

    fig_height = max(3.5, 0.7 * len(table_df) + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(PALETTE["blue"])
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F4F7FB")
        else:
            cell.set_facecolor("white")

    plt.title("Benchmark Comparison Table", fontsize=14, fontweight="bold", pad=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Render benchmark CSV into chart and table images.")
    parser.add_argument(
        "--csv-path",
        help="Optional benchmark CSV path. Defaults to the latest comparison CSV."
    )
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.csv_path)
    comparison_df = build_best_comparison(pd.read_csv(csv_path))

    stem = csv_path.stem
    chart_path = FIGURES_DIR / f"{stem}_chart.png"
    table_path = FIGURES_DIR / f"{stem}_table.png"

    save_chart(comparison_df, chart_path)
    save_table(comparison_df, table_path)

    print(f"Saved {chart_path}")
    print(f"Saved {table_path}")


if __name__ == "__main__":
    main()
