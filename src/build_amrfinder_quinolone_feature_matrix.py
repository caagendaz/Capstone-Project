"""Build a quinolone-focused genotype feature matrix from AMRFinder outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sanitize_feature_name(symbol: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(symbol)).strip("_")
    if not cleaned:
        cleaned = "unknown_feature"
    if cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    return f"feat_{cleaned}"


def resolve_label_columns(labels_df: pd.DataFrame) -> tuple[str, str]:
    sample_candidates = ["Name", "sample_name", "isolate_id", "biosample_accession"]
    label_candidates = ["ciprofloxacin_phenotype", "resistance_phenotype", "resistant_phenotype"]

    sample_col = next((c for c in sample_candidates if c in labels_df.columns), None)
    label_col = next((c for c in label_candidates if c in labels_df.columns), None)

    if sample_col is None:
        raise ValueError("Could not find a sample identifier column in labels file.")
    if label_col is None:
        raise ValueError("Could not find a resistance phenotype column in labels file.")

    return sample_col, label_col


def assign_feature_columns(symbols: list[str]) -> dict[str, str]:
    assigned: dict[str, str] = {}
    used_counts: dict[str, int] = {}

    for symbol in symbols:
        base_name = sanitize_feature_name(symbol)
        suffix_count = used_counts.get(base_name, 0)
        column_name = base_name if suffix_count == 0 else f"{base_name}_{suffix_count + 1}"
        used_counts[base_name] = suffix_count + 1
        assigned[symbol] = column_name

    return assigned


def filter_relevant_hits(hits_df: pd.DataFrame) -> pd.DataFrame:
    symbol_series = hits_df["Element symbol"].fillna("").astype(str)
    relevant_mask = (
        hits_df["Class"].fillna("").str.contains("QUINOLONE", case=False)
        | hits_df["Subclass"].fillna("").str.contains("QUINOLONE", case=False)
        | symbol_series.str.contains(r"qnr|Ib-cr", case=False, regex=True)
    )
    return hits_df.loc[relevant_mask].copy()


def filter_relevant_mutations(mutations_df: pd.DataFrame, hit_symbols: set[str]) -> pd.DataFrame:
    symbol_series = mutations_df["Element symbol"].fillna("").astype(str)
    name_series = mutations_df["Element name"].fillna("").astype(str)

    relevant_mask = (
        mutations_df["Class"].fillna("").str.contains("QUINOLONE", case=False)
        | mutations_df["Subclass"].fillna("").str.contains("QUINOLONE", case=False)
        | name_series.str.contains("quinolone resistant", case=False)
        | symbol_series.isin(hit_symbols)
    )
    filtered = mutations_df.loc[relevant_mask].copy()
    filtered = filtered[~name_series.loc[filtered.index].str.contains(r"\[WILDTYPE\]", regex=True)]
    filtered = filtered[~name_series.loc[filtered.index].str.contains(r"\[UNKNOWN\]", regex=True)]
    return filtered


def build_binary_matrix(rows_df: pd.DataFrame, sample_col: str, symbol_col: str) -> pd.DataFrame:
    matrix = (
        rows_df[[sample_col, symbol_col]]
        .drop_duplicates()
        .assign(value=1)
        .pivot(index=sample_col, columns=symbol_col, values="value")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a quinolone-focused AMRFinder feature matrix.")
    parser.add_argument("--hits-path", required=True, help="Combined AMRFinder hits TSV")
    parser.add_argument("--mutations-path", required=True, help="Combined AMRFinder mutation_all TSV")
    parser.add_argument("--labels-path", required=True, help="CSV with sample labels/accessions")
    parser.add_argument("--output-dir", required=True, help="Directory for output matrices")
    return parser.parse_args()


def load_amrfinder_table(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if path.is_dir():
        frames = []
        for tsv_path in sorted(path.glob("*.tsv")):
            if not tsv_path.exists() or tsv_path.stat().st_size == 0:
                continue
            try:
                df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
            except pd.errors.EmptyDataError:
                continue
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)

    return pd.read_csv(path, sep="\t", low_memory=False)


def main() -> None:
    args = parse_args()

    hits_df = load_amrfinder_table(args.hits_path)
    mutations_df = load_amrfinder_table(args.mutations_path)
    labels_df = pd.read_csv(args.labels_path, low_memory=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_col, label_col = resolve_label_columns(labels_df)
    labels = labels_df.rename(columns={sample_col: "Name", label_col: "resistance_phenotype"}).copy()
    labels = labels.drop_duplicates(subset=["Name"]).copy()

    if (
        (
            "bioproject_accession" not in labels.columns
            or labels["bioproject_accession"].isna().all()
        )
        and "biosample_accession" in labels.columns
    ):
        bioproject_map_path = PROJECT_ROOT / "data/external/biosample_to_assembly.csv"
        if bioproject_map_path.exists():
            bioproject_map = (
                pd.read_csv(
                    bioproject_map_path,
                    usecols=["biosample_accession", "bioproject_accession"],
                    low_memory=False
                )
                .drop_duplicates(subset=["biosample_accession"])
            )
            labels = labels.merge(
                bioproject_map,
                on="biosample_accession",
                how="left"
            )

    relevant_hits = filter_relevant_hits(hits_df)
    hit_symbols = set(relevant_hits["Element symbol"].dropna().astype(str))
    relevant_mutations = filter_relevant_mutations(mutations_df, hit_symbols)

    combined_rows = pd.concat(
        [
            relevant_hits[["Name", "Element symbol"]].dropna(),
            relevant_mutations[["Name", "Element symbol"]].dropna(),
        ],
        ignore_index=True,
    ).drop_duplicates()

    symbol_to_column = assign_feature_columns(sorted(combined_rows["Element symbol"].unique()))
    feature_rows = combined_rows.copy()
    feature_rows["feature_column"] = feature_rows["Element symbol"].map(symbol_to_column)

    matrix = build_binary_matrix(feature_rows, sample_col="Name", symbol_col="feature_column")
    labeled_matrix = labels.merge(matrix, on="Name", how="left")
    feature_cols = [c for c in matrix.columns if c != "Name"]
    labeled_matrix[feature_cols] = labeled_matrix[feature_cols].fillna(0).astype(int)

    feature_dict = pd.DataFrame(
        [{"feature_column": column, "source_symbol": symbol} for symbol, column in symbol_to_column.items()]
    ).sort_values("feature_column")

    summary = pd.DataFrame([
        {
            "n_samples": len(labeled_matrix),
            "n_features": len(feature_cols),
            "n_hit_rows_used": len(relevant_hits),
            "n_mutation_rows_used": len(relevant_mutations),
            "n_unique_symbols": len(symbol_to_column),
            "n_positive_labels": int((labeled_matrix["resistance_phenotype"] == "R").sum()),
            "n_negative_labels": int((labeled_matrix["resistance_phenotype"] == "S").sum()),
        }
    ])

    matrix.to_csv(output_dir / "amrfinder_quinolone_feature_matrix.csv", index=False)
    labeled_matrix.to_csv(output_dir / "amrfinder_quinolone_labeled_feature_matrix.csv", index=False)
    feature_dict.to_csv(output_dir / "amrfinder_quinolone_feature_dictionary.csv", index=False)
    summary.to_csv(output_dir / "amrfinder_quinolone_feature_matrix_summary.csv", index=False)

    print(f"Saved quinolone feature matrix to {output_dir / 'amrfinder_quinolone_feature_matrix.csv'}")
    print(f"Saved labeled matrix to {output_dir / 'amrfinder_quinolone_labeled_feature_matrix.csv'}")
    print(f"Saved feature dictionary to {output_dir / 'amrfinder_quinolone_feature_dictionary.csv'}")
    print(f"Saved summary to {output_dir / 'amrfinder_quinolone_feature_matrix_summary.csv'}")
    print(f"Samples: {len(labeled_matrix)}")
    print(f"Features: {len(feature_cols)}")


if __name__ == "__main__":
    main()
