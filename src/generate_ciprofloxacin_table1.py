"""
Generate a ciprofloxacin-only Table 1 and matching legend text.
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


FIGURES_DIR = config.RESULT_FIGURES_DIR
BENCHMARK_METRICS_DIR = config.BENCHMARK_METRICS_DIR

GENE_RESULTS = BENCHMARK_METRICS_DIR / "results_CIPROFLOXACIN_GENES_EUCAST_20260406_162934.json"

PALETTE = {
    "blue": "#2A6F97",
    "slate": "#5C677D"
}


def load_result_rows(path: Path, n_total: int) -> list[dict]:
    """Load one result JSON into normalized table rows."""
    with path.open() as f:
        data = json.load(f)

    rows = []
    for model_name, payload in data.items():
        if model_name not in {"logistic_regression", "random_forest", "xgboost"}:
            continue

        metrics = payload["metrics"]
        rows.append({
            "Model": model_name.replace("_", " ").title(),
            "n": n_total,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1_score"],
            "ROC-AUC": metrics["roc_auc"]
        })

    return rows


def build_table_df() -> pd.DataFrame:
    """Build the final ciprofloxacin comparison table."""
    X_train_gene, X_test_gene, _, _ = prepare_training_data(
        "CIPROFLOXACIN",
        feature_source="genes",
        testing_standard="EUCAST"
    )

    df = pd.DataFrame(
        load_result_rows(GENE_RESULTS, len(X_train_gene) + len(X_test_gene))
    )
    model_order = ["Logistic Regression", "Random Forest", "Xgboost"]
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
    df = df.sort_values(["Model"]).reset_index(drop=True)

    for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        df[col] = df[col].map(lambda value: f"{value:.3f}")

    return df


def save_table_image(df: pd.DataFrame, output_path: Path) -> Path:
    """Render the comparison table as a PNG."""
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(PALETTE["blue"])
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F4F7FB")
        else:
            cell.set_facecolor("white")

    plt.title(
        "Table 1. Gene-Based Ciprofloxacin Resistance Prediction Performance",
        fontsize=13,
        fontweight="bold",
        pad=14
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def save_legend(output_path: Path) -> Path:
    """Write the matching title and legend text."""
    legend_text = """Table 1. Gene-Based Ciprofloxacin Resistance Prediction Performance

This table summarizes held-out test performance for ciprofloxacin resistance prediction in Escherichia coli using AMR gene presence/absence features from EUCAST-labeled isolates. Three base classifiers were evaluated on the same gene-feature dataset: logistic regression, random forest, and XGBoost. The gene-based ciprofloxacin dataset included 259 isolates total. Among the evaluated models, XGBoost achieved the highest ROC-AUC and F1 score, although overall discrimination remained modest.
"""
    output_path.write_text(legend_text, encoding="utf-8")
    return output_path


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    table_df = build_table_df()
    table_path = FIGURES_DIR / "table1_ciprofloxacin_model_comparison.png"
    legend_path = FIGURES_DIR / "table1_ciprofloxacin_model_comparison_legend.md"

    save_table_image(table_df, table_path)
    save_legend(legend_path)

    print(f"Saved {table_path}")
    print(f"Saved {legend_path}")


if __name__ == "__main__":
    main()
