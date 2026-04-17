"""Evaluate NCBI ciprofloxacin models with leave-one-BioProject-out testing."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.model_training import ModelTrainer


MATRIX_PATH = Path(
    "data/external/amrfinder_batch_cipro_remaining/amrfinder_quinolone_labeled_feature_matrix.csv"
)
RESULTS_DIR = config.NCBI_LAYOUT["root"] / "leave_one_bioproject_out"
METRICS_DIR = config.NCBI_LAYOUT["current_metrics"]
FIGURES_DIR = config.NCBI_LAYOUT["result_figures"]


def load_matrix() -> pd.DataFrame:
    df = pd.read_csv(MATRIX_PATH, low_memory=False)
    df = df[df["antibiotic"].astype(str).str.upper() == "CIPROFLOXACIN"].copy()
    df = df[df["resistance_phenotype"].isin(["R", "S"])].copy()
    df = df.dropna(subset=["bioproject_accession"]).copy()
    return df


def evaluate_fold(train_df: pd.DataFrame,
                  test_df: pd.DataFrame,
                  held_out_project: str,
                  cv: int) -> tuple[list[dict], list[dict]]:
    feature_cols = [col for col in train_df.columns if col.startswith("feat_")]

    X_train = train_df[feature_cols].fillna(0).astype(int)
    X_test = test_df[feature_cols].fillna(0).astype(int)
    y_train = (train_df["resistance_phenotype"].astype(str).str.upper() == "R").astype(int)
    y_test = (test_df["resistance_phenotype"].astype(str).str.upper() == "R").astype(int)

    trainer = ModelTrainer(
        results_dir=str(RESULTS_DIR),
        figures_dir=str(FIGURES_DIR),
        metrics_dir=str(METRICS_DIR),
        use_smote=True,
        use_feature_selection=True,
        n_features=30,
        artifact_prefix=f"CIPROFLOXACIN_NCBI_GENES_LOGO_{held_out_project}"
    )

    X_train_array = X_train.values
    X_test_array = X_test.values
    y_train_array = y_train.values

    models = {
        "logistic_regression": trainer.train_logistic_regression,
        "random_forest": trainer.train_random_forest,
        "xgboost": trainer.train_xgboost,
    }

    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for model_name, train_func in models.items():
        model = train_func(X_train_array, y_train_array, cv=cv)
        metrics = trainer.evaluate_model(model, X_test_array, y_test, model_name)
        y_score = model.predict_proba(X_test_array)[:, 1]
        fold_rows.append({
            "held_out_bioproject": held_out_project,
            "train_n": len(train_df),
            "test_n": len(test_df),
            "test_r": int(y_test.sum()),
            "test_s": int((1 - y_test).sum()),
            "model": model_name,
            **metrics,
        })
        for sample_name, truth, score in zip(test_df["Name"].tolist(), y_test.tolist(), y_score.tolist()):
            prediction_rows.append({
                "held_out_bioproject": held_out_project,
                "model": model_name,
                "Name": sample_name,
                "y_true": int(truth),
                "y_score": float(score),
            })

    return fold_rows, prediction_rows


def save_auc_figure(results_df: pd.DataFrame, output_path: Path) -> None:
    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    bioproject_order = list(dict.fromkeys(results_df["held_out_bioproject"].tolist()))
    model_order = ["logistic_regression", "random_forest", "xgboost"]

    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.22
    x = range(len(bioproject_order))
    offsets = {
        "logistic_regression": -width,
        "random_forest": 0,
        "xgboost": width,
    }
    colors = {
        "logistic_regression": "#1f77b4",
        "random_forest": "#ff7f0e",
        "xgboost": "#2ca02c",
    }

    for model_name in model_order:
        subset = (
            results_df[results_df["model"] == model_name]
            .set_index("held_out_bioproject")
            .reindex(bioproject_order)
        )
        auc_values = subset["roc_auc"].to_numpy()
        bar_positions = [idx + offsets[model_name] for idx in x]
        bars = ax.bar(
            bar_positions,
            auc_values,
            width=width,
            color=colors[model_name],
            label=display_names[model_name]
        )
        for bar, value in zip(bars, auc_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.005,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(bioproject_order, rotation=20, ha="right")
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Leave-One-BioProject-Out ROC-AUC by Held-Out BioProject")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_roc_figure(predictions_df: pd.DataFrame, output_path: Path) -> None:
    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    colors = {
        "logistic_regression": "#1f77b4",
        "random_forest": "#ff7f0e",
        "xgboost": "#2ca02c",
    }
    model_order = ["logistic_regression", "random_forest", "xgboost"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name in model_order:
        model_df = predictions_df[predictions_df["model"] == model_name].copy()
        if model_df.empty or model_df["y_true"].nunique() < 2:
            continue

        fpr, tpr, _ = roc_curve(model_df["y_true"], model_df["y_score"])
        auc = roc_auc_score(model_df["y_true"], model_df["y_score"])
        ax.plot(
            fpr,
            tpr,
            linewidth=2.2,
            color=colors[model_name],
            label=f"{display_names[model_name]} (AUC = {auc:.3f})"
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1.5, label="Random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Leave-One-BioProject-Out ROC Curves\nPooled Held-Out Predictions")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_matrix()
    group_counts = (
        df.groupby("bioproject_accession")["resistance_phenotype"]
        .value_counts()
        .unstack(fill_value=0)
        .sort_index()
    )

    valid_projects = group_counts[(group_counts.get("R", 0) > 0) & (group_counts.get("S", 0) > 0)].index.tolist()
    skipped_projects = group_counts.index.difference(valid_projects).tolist()

    all_rows: list[dict] = []
    all_prediction_rows: list[dict] = []
    for held_out_project in valid_projects:
        test_df = df[df["bioproject_accession"] == held_out_project].copy()
        train_df = df[df["bioproject_accession"] != held_out_project].copy()

        # Use 3-fold CV in the training portion so LOGO can finish in a reasonable time.
        fold_rows, prediction_rows = evaluate_fold(train_df, test_df, held_out_project, cv=3)
        all_rows.extend(fold_rows)
        all_prediction_rows.extend(prediction_rows)

    results_df = pd.DataFrame(all_rows)
    predictions_df = pd.DataFrame(all_prediction_rows)
    summary_df = (
        results_df.groupby("model")[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
        .agg(["mean", "min", "max"])
        .round(4)
    )
    skipped_df = group_counts.loc[skipped_projects].reset_index() if skipped_projects else pd.DataFrame()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = METRICS_DIR / "ncbi_leave_one_bioproject_out_results.csv"
    predictions_path = METRICS_DIR / "ncbi_leave_one_bioproject_out_predictions.csv"
    summary_path = METRICS_DIR / "ncbi_leave_one_bioproject_out_summary.csv"
    skipped_path = METRICS_DIR / "ncbi_leave_one_bioproject_out_skipped_groups.csv"
    figure_path = FIGURES_DIR / "ncbi_leave_one_bioproject_out_auc.png"
    roc_path = FIGURES_DIR / "ncbi_leave_one_bioproject_out_roc.png"

    results_df.to_csv(results_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    summary_df.to_csv(summary_path)
    if not skipped_df.empty:
        skipped_df.to_csv(skipped_path, index=False)
    save_auc_figure(results_df, figure_path)
    save_roc_figure(predictions_df, roc_path)

    print(f"Saved fold-level results to {results_path}")
    print(f"Saved prediction-level results to {predictions_path}")
    print(f"Saved summary results to {summary_path}")
    if not skipped_df.empty:
        print(f"Saved skipped-group summary to {skipped_path}")
    print(f"Saved AUC figure to {figure_path}")
    print(f"Saved ROC figure to {roc_path}")


if __name__ == "__main__":
    main()
