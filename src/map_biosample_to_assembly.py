"""
Map BioSample accessions from an AST export to NCBI assembly accessions.

This script is designed for the NCBI AST Browser / BioSample phenotype exports
used in this project. It reads unique BioSample accessions, looks them up in
NCBI's official assembly summary tables, and writes a mapping file that can be
used for later FASTA download and AMR annotation.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


DEFAULT_OUTPUT = config.EXTERNAL_DATA_DIR / "biosample_to_assembly.csv"


def _read_text_source(path_or_url: str) -> str:
    """Read a local file or URL into text."""
    if path_or_url.startswith(("http://", "https://", "ftp://")):
        response = requests.get(path_or_url, timeout=300)
        response.raise_for_status()
        response.encoding = response.encoding or "utf-8"
        return response.text

    return Path(path_or_url).read_text(encoding="utf-8")


def _load_assembly_summary(path_or_url: str, source_name: str) -> pd.DataFrame:
    """
    Parse an NCBI assembly summary table.

    These files include a header line prefixed with '#', so we extract that
    header explicitly instead of relying on pandas' default parser behavior.
    """
    text = _read_text_source(path_or_url)
    lines = text.splitlines()

    header = None
    data_start = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#assembly_accession") or stripped.startswith("# assembly_accession"):
            header = stripped.lstrip("#").strip().split("\t")
            data_start = idx + 1
            break

    if header is None:
        raise ValueError(f"Could not locate assembly summary header in {path_or_url}")

    data_text = "\n".join(lines[data_start:])
    df = pd.read_csv(io.StringIO(data_text), sep="\t", names=header, dtype=str, low_memory=False)
    df["source_db"] = source_name
    return df


def _select_best_assembly(df: pd.DataFrame) -> pd.DataFrame:
    """Choose one preferred assembly per BioSample accession."""
    work = df.copy()

    work["version_status_rank"] = work["version_status"].fillna("").str.upper().map(
        {"LATEST": 0, "REPLACED": 1, "SUPPRESSED": 2}
    ).fillna(3)
    work["source_rank"] = work["source_db"].map({"RefSeq": 0, "GenBank": 1}).fillna(2)
    work["assembly_level_rank"] = work["assembly_level"].fillna("").str.upper().map(
        {"COMPLETE GENOME": 0, "CHROMOSOME": 1, "SCAFFOLD": 2, "CONTIG": 3}
    ).fillna(4)

    work = work.sort_values(
        ["biosample", "source_rank", "version_status_rank", "assembly_level_rank", "assembly_accession"],
        kind="mergesort"
    )
    best = work.drop_duplicates(subset=["biosample"], keep="first").copy()

    counts = work.groupby("biosample")["assembly_accession"].nunique().rename("assembly_match_count")
    best = best.merge(counts, left_on="biosample", right_index=True, how="left")
    return best.drop(columns=["version_status_rank", "source_rank", "assembly_level_rank"])


def _add_fasta_urls(df: pd.DataFrame) -> pd.DataFrame:
    """Create convenient genome FASTA URLs from the assembly ftp_path field."""
    work = df.copy()
    ftp_path = work["ftp_path"].fillna("")
    basename = ftp_path.str.rstrip("/").str.split("/").str[-1]
    work["genomic_fna_url"] = ftp_path.where(
        ftp_path.eq(""),
        ftp_path.str.rstrip("/") + "/" + basename + "_genomic.fna.gz"
    )
    return work


def _resolve_biosample_column(df: pd.DataFrame, explicit_column: str | None) -> str:
    if explicit_column and explicit_column in df.columns:
        return explicit_column

    for candidate in ("#BioSample", "BioSample", "biosample_accession"):
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not find a BioSample column. Pass --biosample-column explicitly."
    )


def _read_input_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    sep = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, dtype=str, low_memory=False)


def _unique_nonempty(values: Iterable[str]) -> list[str]:
    series = pd.Series(list(values), dtype="string").dropna().str.strip()
    series = series[series.ne("")]
    return sorted(series.unique().tolist())


def build_mapping(
    input_path: str,
    output_path: str,
    biosample_column: str | None = None,
    refseq_source: str = config.NCBI_ASSEMBLY_SUMMARY_REFSEQ_URL,
    genbank_source: str = config.NCBI_ASSEMBLY_SUMMARY_GENBANK_URL,
) -> pd.DataFrame:
    """Build and save a BioSample-to-assembly mapping table."""
    input_df = _read_input_table(input_path)
    biosample_col = _resolve_biosample_column(input_df, biosample_column)
    biosamples = _unique_nonempty(input_df[biosample_col])

    if not biosamples:
        raise ValueError("No BioSample accessions found in the input file")

    refseq_df = _load_assembly_summary(refseq_source, "RefSeq")
    genbank_df = _load_assembly_summary(genbank_source, "GenBank")
    assembly_df = pd.concat([refseq_df, genbank_df], ignore_index=True, sort=False)

    keep_cols = [
        "assembly_accession", "bioproject", "biosample", "wgs_master",
        "refseq_category", "taxid", "species_taxid", "organism_name",
        "infraspecific_name", "isolate", "version_status", "assembly_level",
        "release_type", "genome_rep", "seq_rel_date", "asm_name",
        "submitter", "gbrs_paired_asm", "paired_asm_comp", "ftp_path",
        "source_db"
    ]
    available_cols = [col for col in keep_cols if col in assembly_df.columns]
    assembly_df = assembly_df[available_cols].copy()
    assembly_df["biosample"] = assembly_df["biosample"].astype("string").str.strip()
    assembly_df = assembly_df[assembly_df["biosample"].isin(biosamples)].copy()

    best_matches = _select_best_assembly(assembly_df) if not assembly_df.empty else pd.DataFrame(columns=available_cols)
    best_matches = _add_fasta_urls(best_matches) if not best_matches.empty else best_matches

    requested = pd.DataFrame({"biosample_accession": biosamples})
    if best_matches.empty:
        output_df = requested.copy()
    else:
        rename_map = {"biosample": "biosample_accession", "bioproject": "bioproject_accession"}
        output_df = requested.merge(
            best_matches.rename(columns=rename_map),
            on="biosample_accession",
            how="left"
        )

    output_df["has_assembly"] = output_df["assembly_accession"].notna()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_file, index=False)
    return output_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map BioSample accessions to NCBI assembly accessions")
    parser.add_argument(
        "--input",
        default=str(config.RAW_DATA_DIR / "asts.tsv"),
        help="Input AST TSV/CSV containing BioSample accessions"
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path for the BioSample -> assembly mapping"
    )
    parser.add_argument(
        "--biosample-column",
        default=None,
        help="BioSample column name if auto-detection does not work"
    )
    parser.add_argument(
        "--refseq-source",
        default=config.NCBI_ASSEMBLY_SUMMARY_REFSEQ_URL,
        help="Local path or URL to assembly_summary_refseq.txt"
    )
    parser.add_argument(
        "--genbank-source",
        default=config.NCBI_ASSEMBLY_SUMMARY_GENBANK_URL,
        help="Local path or URL to assembly_summary_genbank.txt"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_df = build_mapping(
        input_path=args.input,
        output_path=args.output,
        biosample_column=args.biosample_column,
        refseq_source=args.refseq_source,
        genbank_source=args.genbank_source
    )

    matched = int(output_df["has_assembly"].sum())
    total = len(output_df)
    print(f"Saved BioSample mapping to {args.output}")
    print(f"Matched assemblies: {matched}/{total} ({matched / total:.1%})")


if __name__ == "__main__":
    main()
