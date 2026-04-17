"""
Generate a figure from the AMRFinder ciprofloxacin pilot feature matrix.
"""

from __future__ import annotations

import argparse
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


DEFAULT_LABELED_MATRIX = (
    config.EXTERNAL_DATA_DIR / "amrfinder_batch_cipro_pilot25" / "amrfinder_labeled_feature_matrix.csv"
)
DEFAULT_FEATURE_DICT = (
    config.EXTERNAL_DATA_DIR / "amrfinder_batch_cipro_pilot25" / "amrfinder_feature_dictionary.csv"
)
DEFAULT_OUTPUT = (
    config.get_results_layout("NCBI")["result_figures"] / "ncbi_pilot" / "amrfinder_ciprofloxacin_pilot25_feature_enrichment.png"
)
DEFAULT_TABLE_OUTPUT = (
    config.EXTERNAL_DATA_DIR / "amrfinder_batch_cipro_pilot25" / "amrfinder_feature_prevalence_summary.csv"
)

PALETTE = {
    "resistant": "#2A6F97",
    "susceptible": "#F4A261",
    "accent": "#5C677D",
}


def compute_prevalence_table(labeled_matrix: pd.DataFrame,
                             feature_dictionary: pd.DataFrame,
                             top_n: int,
                             min_total: int) -> pd.DataFrame:
    """Summarize feature prevalence in resistant vs susceptible isolates."""
    feature_columns = [column for column in labeled_matrix.columns if column.startswith("feat_")]
    long_rows = []

    resistant = labeled_matrix[labeled_matrix["resistance_phenotype"] == "R"].copy()
    susceptible = labeled_matrix[labeled_matrix["resistance_phenotype"] == "S"].copy()

    for feature_column in feature_columns:
        r_count = int(resistant[feature_column].sum())
        s_count = int(susceptible[feature_column].sum())
        total_count = r_count + s_count
        if total_count < min_total:
            continue

        long_rows.append({
            "feature_column": feature_column,
            "r_count": r_count,
            "s_count": s_count,
            "total_count": total_count,
            "r_prevalence": r_count / len(resistant) if len(resistant) else 0.0,
            "s_prevalence": s_count / len(susceptible) if len(susceptible) else 0.0,
        })

    prevalence = pd.DataFrame(long_rows)
    if prevalence.empty:
        return prevalence

    prevalence["prevalence_gap"] = prevalence["r_prevalence"] - prevalence["s_prevalence"]
    prevalence["abs_gap"] = prevalence["prevalence_gap"].abs()

    prevalence = prevalence.merge(feature_dictionary, on="feature_column", how="left")
    prevalence["display_label"] = prevalence["Element symbol"].fillna(prevalence["feature_column"])

    prevalence = prevalence.sort_values(
        ["abs_gap", "r_prevalence", "total_count"],
        ascending=[False, False, False],
    ).head(top_n)

    prevalence = prevalence.sort_values("prevalence_gap", ascending=True).reset_index(drop=True)
    return prevalence


def plot_prevalence(prevalence: pd.DataFrame, output_path: Path) -> None:
    """Plot resistant vs susceptible prevalence for the selected features."""
    fig_height = max(6, 0.45 * len(prevalence) + 2.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))

    y_positions = range(len(prevalence))
    bar_height = 0.36

    ax.barh(
        [y - bar_height / 2 for y in y_positions],
        prevalence["r_prevalence"],
        height=bar_height,
        color=PALETTE["resistant"],
        label="Resistant",
    )
    ax.barh(
        [y + bar_height / 2 for y in y_positions],
        prevalence["s_prevalence"],
        height=bar_height,
        color=PALETTE["susceptible"],
        label="Susceptible",
    )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(prevalence["display_label"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Feature prevalence within class")
    ax.set_title("NCBI Ciprofloxacin Pilot: AMRFinder Feature Enrichment")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False)

    for idx, row in prevalence.iterrows():
        ax.text(
            min(row["r_prevalence"] + 0.02, 0.98),
            idx - bar_height / 2,
            f"{row['r_count']}/{13}",
            va="center",
            ha="left",
            fontsize=9,
            color=PALETTE["accent"],
        )
        ax.text(
            min(row["s_prevalence"] + 0.02, 0.98),
            idx + bar_height / 2,
            f"{row['s_count']}/{12}",
            va="center",
            ha="left",
            fontsize=9,
            color=PALETTE["accent"],
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a feature enrichment figure from the AMRFinder pilot matrix.")
    parser.add_argument("--labeled-matrix", default=str(DEFAULT_LABELED_MATRIX), help="Labeled AMRFinder feature matrix CSV")
    parser.add_argument("--feature-dictionary", default=str(DEFAULT_FEATURE_DICT), help="AMRFinder feature dictionary CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output figure path")
    parser.add_argument("--summary-output", default=str(DEFAULT_TABLE_OUTPUT), help="Output CSV of prevalence summary")
    parser.add_argument("--top-n", type=int, default=12, help="Number of features to plot")
    parser.add_argument("--min-total", type=int, default=2, help="Minimum total count across all samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    labeled_matrix = pd.read_csv(args.labeled_matrix, low_memory=False)
    feature_dictionary = pd.read_csv(args.feature_dictionary, low_memory=False)

    prevalence = compute_prevalence_table(
        labeled_matrix=labeled_matrix,
        feature_dictionary=feature_dictionary,
        top_n=args.top_n,
        min_total=args.min_total,
    )

    if prevalence.empty:
        raise ValueError("No features met the plotting criteria.")

    plot_prevalence(prevalence, Path(args.output))
    prevalence.to_csv(args.summary_output, index=False)

    print(f"Saved figure to {args.output}")
    print(f"Saved prevalence summary to {args.summary_output}")


if __name__ == "__main__":
    main()
