"""
Download genome FASTA files for NCBI assemblies mapped from BioSample IDs.

This script consumes the output of map_biosample_to_assembly.py and downloads
the corresponding genomic FASTA files, optionally restricting the download set
to BioSamples present in a phenotype file.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


DEFAULT_MAPPING = config.EXTERNAL_DATA_DIR / "biosample_to_assembly.csv"
DEFAULT_OUTPUT_DIR = config.NCBI_FASTA_DIR
DEFAULT_MANIFEST = config.EXTERNAL_DATA_DIR / "ncbi_fasta_download_manifest.csv"
DEFAULT_REQUEST_TIMEOUT = 300
DEFAULT_DOWNLOAD_RETRIES = 5
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0


def _resolve_biosample_column(df: pd.DataFrame, explicit_column: str | None = None) -> str:
    if explicit_column and explicit_column in df.columns:
        return explicit_column

    for candidate in ("biosample_accession", "#BioSample", "BioSample"):
        if candidate in df.columns:
            return candidate

    for column in df.columns:
        lower = column.lower()
        if lower.startswith("biosample_accession"):
            return column

    raise ValueError("Could not find a BioSample column in the provided file")


def _read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    sep = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, dtype=str, low_memory=False)


def _unique_nonempty(values: Iterable[str]) -> list[str]:
    series = pd.Series(list(values), dtype="string").dropna().str.strip()
    series = series[series.ne("")]
    return sorted(series.unique().tolist())


def _sanitize_name(value: str) -> str:
    text = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in str(value))
    return text.strip("_") or "unknown"


def _sort_download_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Prioritize higher-quality sources and assemblies before large GenBank contig tails."""
    work = df.copy()
    if "source_db" in work.columns:
        source_rank = {"RefSeq": 0, "GenBank": 1}
        work["_source_rank"] = work["source_db"].map(source_rank).fillna(9)
    else:
        work["_source_rank"] = 9

    if "assembly_level" in work.columns:
        assembly_rank = {"Complete Genome": 0, "Chromosome": 1, "Scaffold": 2, "Contig": 3}
        work["_assembly_rank"] = work["assembly_level"].map(assembly_rank).fillna(9)
    else:
        work["_assembly_rank"] = 9

    work = work.sort_values(
        ["_source_rank", "_assembly_rank", "biosample_accession"],
        kind="mergesort"
    )
    return work.drop(columns=["_source_rank", "_assembly_rank"])


def _download_stream(source: str,
                     destination: Path,
                     *,
                     timeout: int = DEFAULT_REQUEST_TIMEOUT,
                     retries: int = DEFAULT_DOWNLOAD_RETRIES,
                     backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS) -> int:
    if source.startswith(("http://", "https://", "ftp://")):
        session = requests.Session()
        session.headers.update({"User-Agent": "capstone-project-ncbi-downloader/1.0"})
        last_error: Exception | None = None
        temp_destination = destination.with_suffix(destination.suffix + ".part")

        for attempt in range(1, retries + 1):
            try:
                if temp_destination.exists():
                    temp_destination.unlink()

                with session.get(source, stream=True, timeout=timeout) as response:
                    response.raise_for_status()
                    with open(temp_destination, "wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)

                if destination.exists():
                    destination.unlink()
                temp_destination.replace(destination)
                return attempt
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if temp_destination.exists():
                    temp_destination.unlink()
                if attempt < retries:
                    time.sleep(backoff_seconds * attempt)

        raise RuntimeError(f"Download failed after {retries} attempts: {last_error}") from last_error

    local_source = Path(source)
    if not local_source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    shutil.copyfile(local_source, destination)
    return 1


def _extract_gzip(gzip_path: Path, output_path: Path) -> None:
    temp_output = output_path.with_suffix(output_path.suffix + ".part")
    if temp_output.exists():
        temp_output.unlink()
    with gzip.open(gzip_path, "rb") as src, open(temp_output, "wb") as dst:
        shutil.copyfileobj(src, dst)
    if output_path.exists():
        output_path.unlink()
    temp_output.replace(output_path)


def select_download_set(mapping_df: pd.DataFrame,
                        phenotype_file: str | None = None,
                        biosample_column: str | None = None,
                        only_has_assembly: bool = True) -> pd.DataFrame:
    """Optionally restrict the mapping to BioSamples present in a phenotype file."""
    work = mapping_df.copy()
    mapping_biosample_col = _resolve_biosample_column(work)

    if mapping_biosample_col != "biosample_accession":
        work = work.rename(columns={mapping_biosample_col: "biosample_accession"})

    if only_has_assembly and "has_assembly" in work.columns:
        has_assembly = work["has_assembly"].astype(str).str.lower().isin({"true", "1", "yes"})
        work = work[has_assembly].copy()

    if phenotype_file:
        phenotype_df = _read_table(phenotype_file)
        biosample_col = _resolve_biosample_column(phenotype_df, biosample_column)
        biosamples = _unique_nonempty(phenotype_df[biosample_col])
        work = work[work["biosample_accession"].astype("string").isin(biosamples)].copy()

    work = work.dropna(subset=["genomic_fna_url", "assembly_accession", "biosample_accession"])
    return work


def download_fastas(mapping_path: str,
                    output_dir: str,
                    manifest_path: str,
                    phenotype_file: str | None = None,
                    phenotype_biosample_column: str | None = None,
                    limit: int | None = None,
                    extract: bool = True,
                    keep_gzip: bool = False,
                    dry_run: bool = False,
                    skip_existing: bool = True,
                    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
                    retries: int = DEFAULT_DOWNLOAD_RETRIES,
                    backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS) -> pd.DataFrame:
    """Download FASTA files and write a download manifest."""
    mapping_df = pd.read_csv(mapping_path, dtype=str, low_memory=False)
    selected = select_download_set(
        mapping_df,
        phenotype_file=phenotype_file,
        biosample_column=phenotype_biosample_column,
        only_has_assembly=True
    )
    selected = _sort_download_candidates(selected)

    if limit is not None:
        selected = selected.head(limit).copy()

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for row in selected.itertuples(index=False):
        biosample = str(row.biosample_accession)
        assembly = str(row.assembly_accession)
        source_url = str(row.genomic_fna_url)
        base_name = f"{_sanitize_name(biosample)}__{_sanitize_name(assembly)}"
        gzip_path = output_root / f"{base_name}.fna.gz"
        fasta_path = output_root / f"{base_name}.fna"

        final_path = fasta_path if extract else gzip_path
        status = "pending"
        message = ""
        attempts_used = 0

        try:
            if skip_existing and final_path.exists():
                status = "skipped_existing"
                attempts_used = 0
            elif dry_run:
                status = "dry_run"
            else:
                if gzip_path.exists():
                    gzip_path.unlink()
                attempts_used = _download_stream(
                    source_url,
                    gzip_path,
                    timeout=request_timeout,
                    retries=retries,
                    backoff_seconds=backoff_seconds
                )
                if extract:
                    if fasta_path.exists():
                        fasta_path.unlink()
                    _extract_gzip(gzip_path, fasta_path)
                    if not keep_gzip and gzip_path.exists():
                        gzip_path.unlink()
                status = "downloaded"
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            message = str(exc)
            if gzip_path.exists():
                gzip_path.unlink()
            if fasta_path.exists() and final_path == fasta_path:
                fasta_path.unlink()

        manifest_rows.append({
            "biosample_accession": biosample,
            "assembly_accession": assembly,
            "source_db": getattr(row, "source_db", ""),
            "assembly_level": getattr(row, "assembly_level", ""),
            "genomic_fna_url": source_url,
            "output_path": str(final_path),
            "status": status,
            "attempts_used": attempts_used,
            "message": message
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_file, index=False)
    return manifest_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NCBI FASTA files from a BioSample mapping")
    parser.add_argument("--mapping-csv", default=str(DEFAULT_MAPPING), help="BioSample to assembly mapping CSV")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for downloaded FASTA files")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Output CSV manifest for download results")
    parser.add_argument(
        "--phenotype-file",
        default=str(config.RAW_DATA_DIR / config.NCBI_RAW_FILENAME),
        help="Optional phenotype TSV/CSV used to restrict BioSamples before download"
    )
    parser.add_argument(
        "--phenotype-biosample-column",
        default=None,
        help="BioSample column name in the phenotype file if auto-detection fails"
    )
    parser.add_argument("--limit", type=int, default=None, help="Only download the first N matched FASTA files")
    parser.add_argument("--no-extract", action="store_true", help="Keep downloaded .fna.gz files without extracting")
    parser.add_argument("--keep-gzip", action="store_true", help="Keep the .fna.gz file after extraction")
    parser.add_argument("--dry-run", action="store_true", help="Do not download files; only build the manifest")
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist")
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=DEFAULT_DOWNLOAD_RETRIES, help="Number of download retry attempts")
    parser.add_argument("--backoff-seconds", type=float, default=DEFAULT_RETRY_BACKOFF_SECONDS, help="Base backoff between retries")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_df = download_fastas(
        mapping_path=args.mapping_csv,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        phenotype_file=args.phenotype_file,
        phenotype_biosample_column=args.phenotype_biosample_column,
        limit=args.limit,
        extract=not args.no_extract,
        keep_gzip=args.keep_gzip,
        dry_run=args.dry_run,
        skip_existing=not args.force,
        request_timeout=args.request_timeout,
        retries=args.retries,
        backoff_seconds=args.backoff_seconds
    )

    status_counts = manifest_df["status"].value_counts(dropna=False).to_dict() if not manifest_df.empty else {}
    print(f"Saved FASTA download manifest to {args.manifest}")
    print(f"Status counts: {status_counts}")


if __name__ == "__main__":
    main()
