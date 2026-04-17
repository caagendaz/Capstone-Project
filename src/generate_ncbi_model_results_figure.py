"""Generate a simple comparison chart for current NCBI ciprofloxacin results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


METRICS_CSV = (
    config.NCBI_LAYOUT["current_metrics"] / "model_comparison_CIPROFLOXACIN_NCBI_GENES.csv"
)
OUTPUT_FIGURE = (
    config.NCBI_LAYOUT["result_figures"] / "ncbi_ciprofloxacin_model_comparison_summary.png"
)

PALETTE = {
    "roc_auc": "#1f77b4",
    "f1_score": "#e67e22",
}


def main() -> None:
    df = pd.read_csv(METRICS_CSV)
    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "stacking_ensemble": "Stacking Ensemble",
        "voting_ensemble": "Voting Ensemble",
    }
    df["display_model"] = df["model"].map(display_names).fillna(df["model"])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df))
    width = 0.36

    roc_positions = [i - width / 2 for i in x]
    f1_positions = [i + width / 2 for i in x]

    ax.bar(
        roc_positions,
        df["roc_auc"],
        width=width,
        color=PALETTE["roc_auc"],
        label="ROC-AUC",
    )
    ax.bar(
        f1_positions,
        df["f1_score"],
        width=width,
        color=PALETTE["f1_score"],
        label="F1",
    )

    for pos, value in zip(roc_positions, df["roc_auc"]):
        ax.text(pos, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for pos, value in zip(f1_positions, df["f1_score"]):
        ax.text(pos, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["display_model"], rotation=20, ha="right")
    ax.set_ylim(0.9, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("NCBI Ciprofloxacin Model Performance")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
