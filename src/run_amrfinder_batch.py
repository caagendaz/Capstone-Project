"""
Run AMRFinderPlus in WSL across a batch of downloaded FASTA files.

This script reads the FASTA download manifest produced by
download_ncbi_fastas.py, runs AMRFinderPlus for each downloaded genome, and
combines the per-sample outputs into project-level TSV files.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


DEFAULT_MANIFEST = config.EXTERNAL_DATA_DIR / "ncbi_fasta_download_manifest_limit25.csv"
DEFAULT_OUTPUT_DIR = config.EXTERNAL_DATA_DIR / "amrfinder_batch"
DEFAULT_CONDA_ROOT = "/home/cagns/miniconda3"
DEFAULT_ENV_NAME = "amrfinder-bioconda"


def windows_to_wsl_path(path: Path) -> str:
    """Convert a Windows path into a WSL /mnt path."""
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    parts = [part.replace("\\", "/") for part in resolved.parts[1:]]
    return f"/mnt/{drive}/" + "/".join(parts)


def run_command(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def combine_outputs(source_dir: Path, combined_path: Path) -> int:
    """Combine all non-empty TSV outputs from a directory into one TSV."""
    frames = []
    for tsv_path in sorted(source_dir.glob("*.tsv")):
        if not tsv_path.exists() or tsv_path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined.to_csv(combined_path, sep="\t", index=False)
        return len(combined)

    pd.DataFrame().to_csv(combined_path, sep="\t", index=False)
    return 0


def run_batch(manifest_path: str,
              output_dir: str,
              conda_root: str,
              env_name: str,
              organism: str = "Escherichia",
              limit: int | None = None,
              overwrite: bool = False,
              mutation_all: bool = True) -> pd.DataFrame:
    """Run AMRFinderPlus across the FASTA files listed in a manifest."""
    manifest = pd.read_csv(manifest_path, low_memory=False)
    manifest = manifest[manifest["status"].isin(["downloaded", "skipped_existing"])].copy()

    if limit is not None:
        manifest = manifest.head(limit).copy()

    base_dir = Path(output_dir)
    hits_dir = base_dir / "hits"
    mutations_dir = base_dir / "mutation_all"
    logs_dir = base_dir / "logs"
    for directory in (base_dir, hits_dir, mutations_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    conda_sh = f"{conda_root}/etc/profile.d/conda.sh"

    for record in manifest.itertuples(index=False):
        biosample = str(record.biosample_accession)
        fasta_path = Path(str(record.output_path))

        hits_path = hits_dir / f"{biosample}.tsv"
        mutation_path = mutations_dir / f"{biosample}.tsv"
        log_path = logs_dir / f"{biosample}.log"

        if not overwrite and hits_path.exists():
            status = "skipped_existing"
            returncode = 0
            stdout = ""
            stderr = ""
        else:
            fasta_wsl = windows_to_wsl_path(fasta_path)
            hits_wsl = windows_to_wsl_path(hits_path)
            log_wsl = windows_to_wsl_path(log_path)

            amrfinder_parts = [
                "amrfinder",
                f"--nucleotide {fasta_wsl}",
                f"--organism {organism}",
                f"--name {biosample}",
                f"--output {hits_wsl}",
                f"--log {log_wsl}",
            ]

            if mutation_all:
                mutation_wsl = windows_to_wsl_path(mutation_path)
                amrfinder_parts.append(f"--mutation_all {mutation_wsl}")

            bash_command = " && ".join([
                f"source {conda_sh}",
                f"conda activate {env_name}",
                " ".join(amrfinder_parts),
            ])
            result = run_command(["wsl", "-e", "bash", "-lc", bash_command])
            status = "completed" if result.returncode == 0 else "failed"
            returncode = result.returncode
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

        rows.append({
            "biosample_accession": biosample,
            "assembly_accession": getattr(record, "assembly_accession", ""),
            "fasta_path": str(fasta_path),
            "hits_path": str(hits_path),
            "mutation_all_path": str(mutation_path),
            "log_path": str(log_path),
            "status": status,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
        })

    run_manifest = pd.DataFrame(rows)
    run_manifest.to_csv(base_dir / "run_manifest.csv", index=False)

    hits_rows = combine_outputs(hits_dir, base_dir / "amrfinder_hits_combined.tsv")
    mutation_rows = 0
    if mutation_all:
        mutation_rows = combine_outputs(mutations_dir, base_dir / "amrfinder_mutation_all_combined.tsv")

    summary = pd.DataFrame([{
        "samples_requested": len(manifest),
        "samples_completed": int((run_manifest["status"] == "completed").sum()),
        "samples_failed": int((run_manifest["status"] == "failed").sum()),
        "samples_skipped_existing": int((run_manifest["status"] == "skipped_existing").sum()),
        "combined_hits_rows": hits_rows,
        "combined_mutation_rows": mutation_rows,
    }])
    summary.to_csv(base_dir / "summary.csv", index=False)

    return run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMRFinderPlus in WSL across a batch of FASTA files")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="FASTA download manifest CSV")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for AMRFinder outputs")
    parser.add_argument("--conda-root", default=DEFAULT_CONDA_ROOT, help="Conda root inside WSL")
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME, help="Conda environment name inside WSL")
    parser.add_argument("--organism", default="Escherichia", help="AMRFinder organism setting")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N downloaded samples")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-sample outputs")
    parser.add_argument("--no-mutation-all", action="store_true", help="Skip mutation_all output files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_manifest = run_batch(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        conda_root=args.conda_root,
        env_name=args.env_name,
        organism=args.organism,
        limit=args.limit,
        overwrite=args.overwrite,
        mutation_all=not args.no_mutation_all,
    )
    print(f"Saved batch outputs to {args.output_dir}")
    print(f"Status counts: {run_manifest['status'].value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    main()
