"""Build an accession table for completed NCBI AMRFinder isolates."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHENOTYPE_CSV = PROJECT_ROOT / "data" / "processed" / "cleaned_resistance_data_ncbi.csv"
ASSEMBLY_CSV = PROJECT_ROOT / "data" / "external" / "biosample_to_assembly.csv"
HITS_DIR = PROJECT_ROOT / "data" / "external" / "amrfinder_batch_cipro_remaining" / "hits"
OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "ciprofloxacin_ncbi_completed_1000_accession_table.csv"
)
OUTPUT_MD = (
    PROJECT_ROOT
    / "results"
    / "NCBI"
    / "metrics"
    / "ciprofloxacin_ncbi_completed_1000_accession_table.md"
)


def _write_markdown(df: pd.DataFrame, output_path: Path) -> None:
    headers = list(df.columns)
    lines = [
        "# NCBI Ciprofloxacin Completed AMRFinder Accession Table",
        "",
        f"Total completed isolates: {len(df)}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.fillna("").itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    phenotype_df = pd.read_csv(PHENOTYPE_CSV)
    phenotype_df = phenotype_df[
        (phenotype_df["antibiotic"] == "CIPROFLOXACIN")
        & (phenotype_df["resistance_phenotype"].isin(["R", "S"]))
    ].copy()

    assembly_df = pd.read_csv(ASSEMBLY_CSV)
    assembly_df = assembly_df[
        ["biosample_accession", "assembly_accession", "assembly_level", "source_db"]
    ].copy()

    completed_hits_df = pd.DataFrame(
        {"biosample_accession": sorted(path.stem for path in HITS_DIR.glob("*.tsv"))}
    )

    merged_df = phenotype_df.merge(
        assembly_df, on="biosample_accession", how="left"
    ).merge(completed_hits_df, on="biosample_accession", how="inner")

    phenotype_order = {"R": 0, "S": 1, "I": 2}
    merged_df["_phenotype_order"] = (
        merged_df["resistant_phenotype"].map(phenotype_order).fillna(99)
    )

    output_df = merged_df[
        [
            "ncbi_isolate_accession",
            "isolate_id",
            "biosample_accession",
            "assembly_accession",
            "antibiotic",
            "resistant_phenotype",
            "testing_standard",
            "mic_value",
            "measurement_unit",
            "assembly_level",
            "source_db",
            "location",
            "host",
            "isolation_source",
        ]
    ].rename(
        columns={
            "isolate_id": "sample_name",
            "resistant_phenotype": "ciprofloxacin_phenotype",
            "mic_value": "ciprofloxacin_mic",
        }
    )

    output_df["_phenotype_order"] = merged_df["_phenotype_order"].values
    output_df = output_df.sort_values(
        by=["_phenotype_order", "biosample_accession", "assembly_accession"],
        ascending=[True, True, True],
    ).drop(columns=["_phenotype_order"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)
    _write_markdown(output_df, OUTPUT_MD)

    print(f"Wrote CSV: {OUTPUT_CSV}")
    print(f"Wrote Markdown: {OUTPUT_MD}")
    print(f"Rows: {len(output_df)}")


if __name__ == "__main__":
    main()
