"""
Build a pilot genotype feature matrix from AMRFinderPlus batch outputs.

This script converts the combined AMRFinder hits table into a binary
sample-by-feature matrix and joins it to phenotype labels for downstream
inspection or modeling.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


DEFAULT_HITS_PATH = config.EXTERNAL_DATA_DIR / "amrfinder_batch_cipro_pilot25" / "amrfinder_hits_combined.tsv"
DEFAULT_LABELS_PATH = config.EXTERNAL_DATA_DIR / "ciprofloxacin_ncbi_pilot25_mapping.csv"
DEFAULT_OUTPUT_DIR = config.EXTERNAL_DATA_DIR / "amrfinder_batch_cipro_pilot25"


def sanitize_feature_name(symbol: str) -> str:
    """Convert an AMRFinder element symbol into a stable CSV-friendly column."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(symbol)).strip("_")
    if not cleaned:
        cleaned = "unknown_feature"
    if cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    return f"feat_{cleaned}"


def resolve_sample_column(df: pd.DataFrame) -> str:
    """Find the sample identifier column in the labels table."""
    candidates = [
        "isolate_id",
        "biosample_accession",
        "biosample_accession_x",
        "biosample_accession_y",
        "Name",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError("Could not find a sample identifier column in the labels table.")


def assign_feature_columns(symbols: list[str]) -> dict[str, str]:
    """Create unique cleaned column names for each AMRFinder symbol."""
    assigned: dict[str, str] = {}
    used_counts: dict[str, int] = {}

    for symbol in symbols:
        base_name = sanitize_feature_name(symbol)
        suffix_count = used_counts.get(base_name, 0)
        if suffix_count == 0:
            column_name = base_name
        else:
            column_name = f"{base_name}_{suffix_count + 1}"
        used_counts[base_name] = suffix_count + 1
        assigned[symbol] = column_name

    return assigned


def build_feature_dictionary(hits_df: pd.DataFrame,
                             symbol_to_column: dict[str, str]) -> pd.DataFrame:
    """Create a symbol-to-column mapping table with lightweight metadata."""
    feature_dict = (
        hits_df[
            [
                "Element symbol",
                "Element name",
                "Type",
                "Subtype",
                "Class",
                "Subclass",
                "Scope",
            ]
        ]
        .drop_duplicates()
        .sort_values(["Class", "Subclass", "Element symbol"], na_position="last")
        .copy()
    )
    feature_dict["feature_column"] = feature_dict["Element symbol"].map(symbol_to_column)
    return feature_dict[
        [
            "feature_column",
            "Element symbol",
            "Element name",
            "Type",
            "Subtype",
            "Class",
            "Subclass",
            "Scope",
        ]
    ]


def build_matrix(hits_df: pd.DataFrame,
                 symbol_to_column: dict[str, str]) -> pd.DataFrame:
    """Pivot AMRFinder hits into a binary feature matrix keyed by sample."""
    matrix = (
        hits_df.assign(value=1)
        .pivot_table(
            index="Name",
            columns="Element symbol",
            values="value",
            aggfunc="max",
            fill_value=0,
        )
        .astype(int)
        .reset_index()
    )

    renamed_columns = {
        column: symbol_to_column[column]
        for column in matrix.columns
        if column != "Name"
    }
    matrix = matrix.rename(columns=renamed_columns)
    return matrix


def prepare_labels(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce the phenotype mapping to one row per sample."""
    sample_column = resolve_sample_column(labels_df)
    labels = labels_df.copy()
    labels = labels.rename(columns={sample_column: "Name"})
    labels = labels.drop_duplicates(subset=["Name"]).copy()
    return labels


def save_summary(summary_path: Path,
                 labeled_matrix: pd.DataFrame,
                 feature_dict: pd.DataFrame,
                 hits_df: pd.DataFrame) -> None:
    """Save a small one-row summary CSV for quick inspection."""
    feature_columns = [column for column in labeled_matrix.columns if column.startswith("feat_")]
    summary = pd.DataFrame([{
        "n_samples": len(labeled_matrix),
        "n_features": len(feature_columns),
        "n_positive_labels": int((labeled_matrix["resistance_phenotype"] == "R").sum())
        if "resistance_phenotype" in labeled_matrix.columns else None,
        "n_negative_labels": int((labeled_matrix["resistance_phenotype"] == "S").sum())
        if "resistance_phenotype" in labeled_matrix.columns else None,
        "n_amrfinder_hits": len(hits_df),
        "n_unique_symbols": feature_dict["Element symbol"].nunique(),
        "n_quinolone_features": int(feature_dict["Class"].astype(str).str.contains("QUINOLONE", na=False).sum()),
    }])
    summary.to_csv(summary_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a binary feature matrix from AMRFinderPlus hits.")
    parser.add_argument("--hits-path", default=str(DEFAULT_HITS_PATH), help="Combined AMRFinder hits TSV")
    parser.add_argument("--labels-path", default=str(DEFAULT_LABELS_PATH), help="Pilot label/mapping CSV")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for matrix outputs")
    parser.add_argument(
        "--include-non-amr",
        action="store_true",
        help="Keep non-AMR hits such as stress features instead of filtering to Type == AMR",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hits_path = Path(args.hits_path)
    labels_path = Path(args.labels_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hits_df = pd.read_csv(hits_path, sep="\t", low_memory=False)
    labels_df = pd.read_csv(labels_path, low_memory=False)

    if not args.include_non_amr and "Type" in hits_df.columns:
        hits_df = hits_df[hits_df["Type"] == "AMR"].copy()

    hits_df = hits_df.dropna(subset=["Name", "Element symbol"]).copy()
    labels = prepare_labels(labels_df)
    symbol_to_column = assign_feature_columns(sorted(hits_df["Element symbol"].unique()))
    feature_dict = build_feature_dictionary(hits_df, symbol_to_column)
    feature_matrix = build_matrix(hits_df, symbol_to_column)

    labeled_matrix = labels.merge(feature_matrix, on="Name", how="left")
    feature_columns = [column for column in feature_matrix.columns if column != "Name"]
    labeled_matrix[feature_columns] = labeled_matrix[feature_columns].fillna(0).astype(int)

    feature_matrix_path = output_dir / "amrfinder_feature_matrix.csv"
    labeled_matrix_path = output_dir / "amrfinder_labeled_feature_matrix.csv"
    feature_dict_path = output_dir / "amrfinder_feature_dictionary.csv"
    summary_path = output_dir / "amrfinder_feature_matrix_summary.csv"

    feature_matrix.to_csv(feature_matrix_path, index=False)
    labeled_matrix.to_csv(labeled_matrix_path, index=False)
    feature_dict.to_csv(feature_dict_path, index=False)
    save_summary(summary_path, labeled_matrix, feature_dict, hits_df)

    print(f"Saved feature matrix to {feature_matrix_path}")
    print(f"Saved labeled matrix to {labeled_matrix_path}")
    print(f"Saved feature dictionary to {feature_dict_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Samples: {len(labeled_matrix)}")
    print(f"Features: {len(feature_columns)}")


if __name__ == "__main__":
    main()
