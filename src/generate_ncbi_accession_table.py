"""Generate a clean accession table for the NCBI ciprofloxacin pilot cohort."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "external" / "ciprofloxacin_ncbi_pilot25_mapping.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "external" / "ciprofloxacin_ncbi_pilot25_accession_table.csv"
OUTPUT_MD = (
    PROJECT_ROOT
    / "results"
    / "NCBI"
    / "metrics"
    / "ciprofloxacin_ncbi_pilot25_accession_table.md"
)


KEEP_COLUMNS = [
    "genome_id",
    "isolate_id",
    "biosample_accession_x",
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

RENAME_COLUMNS = {
    "genome_id": "ncbi_isolate_accession",
    "isolate_id": "sample_name",
    "biosample_accession_x": "biosample_accession",
    "resistant_phenotype": "ciprofloxacin_phenotype",
    "mic_value": "ciprofloxacin_mic",
}


def build_accession_table(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    accession_df = df[KEEP_COLUMNS].rename(columns=RENAME_COLUMNS).copy()
    accession_df["ciprofloxacin_mic"] = accession_df["ciprofloxacin_mic"].fillna("")
    phenotype_order = {"R": 0, "S": 1, "I": 2}
    accession_df["_phenotype_order"] = accession_df["ciprofloxacin_phenotype"].map(
        phenotype_order
    ).fillna(99)
    accession_df = accession_df.sort_values(
        by=["_phenotype_order", "biosample_accession", "assembly_accession"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    accession_df = accession_df.drop(columns=["_phenotype_order"])
    return accession_df


def write_markdown_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(df.columns)
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = []
    for row in df.fillna("").itertuples(index=False, name=None):
        body_rows.append("| " + " | ".join(str(value) for value in row) + " |")
    lines = [
        "# NCBI Ciprofloxacin Pilot 25 Accession Table",
        "",
        f"Total isolates: {len(df)}",
        "",
        header_row,
        separator_row,
        *body_rows,
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    accession_df = build_accession_table(INPUT_CSV)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    accession_df.to_csv(OUTPUT_CSV, index=False)
    write_markdown_table(accession_df, OUTPUT_MD)
    print(f"Wrote CSV: {OUTPUT_CSV}")
    print(f"Wrote Markdown: {OUTPUT_MD}")
    print(f"Rows: {len(accession_df)}")


if __name__ == "__main__":
    main()
