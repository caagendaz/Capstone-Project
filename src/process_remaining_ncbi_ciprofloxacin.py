"""
Process the remaining NCBI ciprofloxacin cohort after the 25-sample pilot.

This script:
1. Selects NCBI ciprofloxacin isolates with R/S labels and mapped assemblies.
2. Excludes the pilot BioSamples (and optionally any already-completed batch outputs).
3. Downloads the corresponding FASTA files.
4. Runs AMRFinderPlus in WSL across the downloaded cohort.

It is designed to be resumable. Re-running the script will skip existing
downloads and existing AMRFinder outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.download_ncbi_fastas import download_fastas
from src.run_amrfinder_batch import run_batch


DEFAULT_CLEANED_NCBI = config.PROCESSED_DATA_DIR / "cleaned_resistance_data_ncbi.csv"
DEFAULT_MAPPING = config.EXTERNAL_DATA_DIR / "biosample_to_assembly.csv"
DEFAULT_PILOT = config.EXTERNAL_DATA_DIR / "ciprofloxacin_ncbi_pilot25_mapping.csv"

DEFAULT_COHORT_DIR = config.EXTERNAL_DATA_DIR / "ncbi_ciprofloxacin_remaining"
DEFAULT_FASTA_DIR = config.EXTERNAL_DATA_DIR / "ncbi_fastas_cipro_remaining"
DEFAULT_DOWNLOAD_MANIFEST = config.EXTERNAL_DATA_DIR / "ncbi_fasta_download_manifest_cipro_remaining.csv"
DEFAULT_AMRFINDER_DIR = config.EXTERNAL_DATA_DIR / "amrfinder_batch_cipro_remaining"
DEFAULT_REQUEST_TIMEOUT = 300
DEFAULT_DOWNLOAD_RETRIES = 5
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0


def _resolve_biosample_column(df: pd.DataFrame) -> str:
    for candidate in ("biosample_accession", "biosample_accession_x", "biosample_accession_y", "#BioSample", "BioSample"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Could not find a BioSample column.")


def _normalize_has_assembly(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def build_remaining_cohort(cleaned_ncbi_path: Path,
                           mapping_path: Path,
                           pilot_path: Path,
                           cohort_dir: Path,
                           exclude_completed_dir: Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Create a mapping table for the remaining ciprofloxacin cohort."""
    cleaned = pd.read_csv(cleaned_ncbi_path, low_memory=False)
    mapping = pd.read_csv(mapping_path, low_memory=False)
    pilot = pd.read_csv(pilot_path, low_memory=False) if pilot_path.exists() else pd.DataFrame()

    cleaned = cleaned[cleaned["antibiotic"].astype(str).str.upper() == "CIPROFLOXACIN"].copy()
    cleaned = cleaned[cleaned["resistance_phenotype"].astype(str).str.upper().isin(["R", "S"])].copy()

    cleaned_biosample_col = _resolve_biosample_column(cleaned)
    mapping_biosample_col = _resolve_biosample_column(mapping)
    pilot_biosample_col = _resolve_biosample_column(pilot) if not pilot.empty else None

    if cleaned_biosample_col != "biosample_accession":
        cleaned = cleaned.rename(columns={cleaned_biosample_col: "biosample_accession"})
    if mapping_biosample_col != "biosample_accession":
        mapping = mapping.rename(columns={mapping_biosample_col: "biosample_accession"})
    if pilot_biosample_col and pilot_biosample_col != "biosample_accession":
        pilot = pilot.rename(columns={pilot_biosample_col: "biosample_accession"})

    if "has_assembly" in mapping.columns:
        mapping = mapping[_normalize_has_assembly(mapping["has_assembly"])].copy()

    mapping_cols = [
        "biosample_accession",
        "assembly_accession",
        "genomic_fna_url",
        "assembly_level",
        "source_db",
        "has_assembly",
    ]
    available_mapping_cols = [col for col in mapping_cols if col in mapping.columns]
    merged = cleaned.merge(
        mapping[available_mapping_cols].drop_duplicates(subset=["biosample_accession"]),
        on="biosample_accession",
        how="inner"
    )

    pilot_biosamples = set()
    if not pilot.empty and "biosample_accession" in pilot.columns:
        pilot_biosamples = set(pilot["biosample_accession"].dropna().astype(str))

    completed_biosamples = set()
    if exclude_completed_dir:
        hits_dir = exclude_completed_dir / "hits"
        if hits_dir.exists():
            completed_biosamples.update(path.stem for path in hits_dir.glob("*.tsv"))

    merged = merged[~merged["biosample_accession"].astype(str).isin(pilot_biosamples | completed_biosamples)].copy()

    id_col = "biosample_accession"
    source_rank = {"RefSeq": 0, "GenBank": 1}
    assembly_rank = {"Complete Genome": 0, "Chromosome": 1, "Scaffold": 2, "Contig": 3}
    merged["_source_rank"] = merged.get("source_db", pd.Series(index=merged.index, dtype="object")).map(source_rank).fillna(9)
    merged["_assembly_rank"] = merged.get("assembly_level", pd.Series(index=merged.index, dtype="object")).map(assembly_rank).fillna(9)
    merged = merged.sort_values(["_source_rank", "_assembly_rank", id_col], kind="mergesort")
    merged = merged.drop_duplicates(subset=[id_col], keep="first").drop(columns=["_source_rank", "_assembly_rank"])

    cohort_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = cohort_dir / "ciprofloxacin_ncbi_remaining_mapping.csv"
    merged.to_csv(cohort_path, index=False)
    return merged, cohort_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process the remaining NCBI ciprofloxacin cohort.")
    parser.add_argument("--cleaned-ncbi", default=str(DEFAULT_CLEANED_NCBI), help="Cleaned NCBI phenotype CSV")
    parser.add_argument("--mapping-csv", default=str(DEFAULT_MAPPING), help="BioSample to assembly mapping CSV")
    parser.add_argument("--pilot-csv", default=str(DEFAULT_PILOT), help="Pilot cohort mapping CSV to exclude")
    parser.add_argument("--cohort-dir", default=str(DEFAULT_COHORT_DIR), help="Directory to save the remaining cohort mapping")
    parser.add_argument("--fasta-dir", default=str(DEFAULT_FASTA_DIR), help="Directory for FASTA downloads")
    parser.add_argument("--download-manifest", default=str(DEFAULT_DOWNLOAD_MANIFEST), help="FASTA download manifest path")
    parser.add_argument("--amrfinder-dir", default=str(DEFAULT_AMRFINDER_DIR), help="AMRFinder output directory")
    parser.add_argument("--exclude-existing-batch", default=None, help="Optional AMRFinder batch directory to exclude completed BioSamples from")
    parser.add_argument("--download-limit", type=int, default=None, help="Only download/process the first N remaining isolates")
    parser.add_argument("--process-limit", type=int, default=None, help="Only run AMRFinder on the first N downloaded isolates")
    parser.add_argument("--skip-download", action="store_true", help="Skip the FASTA download step")
    parser.add_argument("--skip-amrfinder", action="store_true", help="Skip the AMRFinder step")
    parser.add_argument("--force-download", action="store_true", help="Redownload FASTAs even if they already exist")
    parser.add_argument("--overwrite-amrfinder", action="store_true", help="Overwrite existing AMRFinder per-sample outputs")
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=DEFAULT_DOWNLOAD_RETRIES, help="Number of retry attempts per download")
    parser.add_argument("--backoff-seconds", type=float, default=DEFAULT_RETRY_BACKOFF_SECONDS, help="Base retry backoff in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cleaned_ncbi_path = Path(args.cleaned_ncbi)
    mapping_path = Path(args.mapping_csv)
    pilot_path = Path(args.pilot_csv)
    cohort_dir = Path(args.cohort_dir)
    fasta_dir = Path(args.fasta_dir)
    download_manifest = Path(args.download_manifest)
    amrfinder_dir = Path(args.amrfinder_dir)
    exclude_existing_batch = Path(args.exclude_existing_batch) if args.exclude_existing_batch else amrfinder_dir

    cohort_df, cohort_path = build_remaining_cohort(
        cleaned_ncbi_path=cleaned_ncbi_path,
        mapping_path=mapping_path,
        pilot_path=pilot_path,
        cohort_dir=cohort_dir,
        exclude_completed_dir=exclude_existing_batch
    )

    print(f"Saved remaining cohort mapping to {cohort_path}")
    print(f"Remaining cohort size: {len(cohort_df)}")

    if not args.skip_download:
        manifest_df = download_fastas(
            mapping_path=str(cohort_path),
            output_dir=str(fasta_dir),
            manifest_path=str(download_manifest),
            phenotype_file=None,
            phenotype_biosample_column=None,
            limit=args.download_limit,
            extract=True,
            keep_gzip=False,
            dry_run=False,
            skip_existing=not args.force_download,
            request_timeout=args.request_timeout,
            retries=args.retries,
            backoff_seconds=args.backoff_seconds
        )
        print(f"Saved FASTA manifest to {download_manifest}")
        print(f"Download status counts: {manifest_df['status'].value_counts(dropna=False).to_dict()}")

    if not args.skip_amrfinder:
        run_manifest = run_batch(
            manifest_path=str(download_manifest),
            output_dir=str(amrfinder_dir),
            conda_root="/home/cagns/miniconda3",
            env_name="amrfinder-bioconda",
            organism="Escherichia",
            limit=args.process_limit,
            overwrite=args.overwrite_amrfinder,
            mutation_all=True
        )
        print(f"Saved AMRFinder batch outputs to {amrfinder_dir}")
        print(f"AMRFinder status counts: {run_manifest['status'].value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    main()
