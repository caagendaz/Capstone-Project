# Capstone Project

Machine learning pipeline for predicting *Escherichia coli* antibiotic resistance, with support for both the legacy BVBRC workflow and the newer NCBI AST plus AMRFinder genotype workflow.

## What This Repo Does

- acquires phenotype data from BVBRC and imported NCBI AST exports
- preprocesses resistance data with stricter deduplication and grouped splits
- trains logistic regression, random forest, XGBoost, and ensemble models
- generates benchmark tables and figures
- builds NCBI genotype features from AMRFinderPlus outputs

## Current Project Layout

```text
capstone_project/
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- external/
|-- models/
|-- results/
|   |-- BVBRC/
|   `-- NCBI/
|-- src/
|-- config.py
|-- main.py
`-- requirements.txt
```

## Data Sources

### BVBRC

Used for the original phenotype and AMR gene workflow.

- phenotype endpoint: `genome_amr`
- organism: *E. coli* (`taxon_id=562`)
- gene annotations: CARD and NDARO

### NCBI

Used for the newer ciprofloxacin-focused expansion.

- phenotype source: NCBI Pathogen Detection AST Browser export
- isolate linking: BioSample to Assembly mapping
- genotype calling: AMRFinderPlus on downloaded genome FASTAs

## Main Entry Points

### Run the standard pipeline

```powershell
python main.py --steps acquire preprocess train validate --antibiotic CIPROFLOXACIN
```

### Import an NCBI AST export

```powershell
python main.py --steps acquire --acquisition-source ncbi --ncbi-ast-file path\to\asts.tsv
```

### Train against the NCBI AMRFinder pilot matrix

```powershell
python main.py --steps train --antibiotic CIPROFLOXACIN --feature-source ncbi_genes --results-source NCBI
```

## NCBI AMRFinder Workflow

The NCBI side of the project is built in stages:

1. import AST rows from NCBI
2. map BioSample accessions to assemblies
3. download genome FASTA files
4. run AMRFinderPlus
5. convert AMRFinder hits into a labeled feature matrix
6. train the same downstream machine learning models

Useful scripts:

- `src/map_biosample_to_assembly.py`
- `src/download_ncbi_fastas.py`
- `src/run_amrfinder_batch.py`
- `src/process_remaining_ncbi_ciprofloxacin.py`
- `src/build_amrfinder_feature_matrix.py`
- `src/generate_amrfinder_pilot_figure.py`

## Results Organization

Legacy and BVBRC outputs live under:

- `results/BVBRC/figures`
- `results/BVBRC/metrics`

NCBI outputs live under:

- `results/NCBI/figures`
- `results/NCBI`

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For AMRFinderPlus, use the WSL/Bioconda environment created during the NCBI workflow.

## Notes

- Large downloaded FASTA folders and long-running AMRFinder batch folders are intentionally ignored in Git.
- The repo keeps curated pilot outputs and figures, not the full genome download cache.
- Current work is focused on ciprofloxacin as the main antibiotic of interest.
