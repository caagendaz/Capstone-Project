"""
Configuration file for E. coli Antibiotic Resistance Project
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
NCBI_FASTA_DIR = EXTERNAL_DATA_DIR / "ncbi_fastas"

# Model directory
MODEL_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"
BVBRC_RESULTS_DIR = RESULTS_DIR / "BVBRC"
NCBI_RESULTS_DIR = RESULTS_DIR / "NCBI"


def build_results_layout(base_dir: Path) -> dict[str, Path]:
    """Build a standard results directory layout rooted at base_dir."""
    figures_dir = base_dir / "figures"
    metrics_dir = base_dir / "metrics"
    return {
        "root": base_dir,
        "figures": figures_dir,
        "result_figures": figures_dir / "results",
        "validation_figures": figures_dir / "validation",
        "legacy_figures": figures_dir / "legacy",
        "metrics": metrics_dir,
        "current_metrics": metrics_dir / "current",
        "benchmark_metrics": metrics_dir / "benchmark",
        "archive_metrics": metrics_dir / "archive",
    }


def get_results_layout(source: str = "BVBRC") -> dict[str, Path]:
    """Return the results layout for a named source."""
    source_key = source.strip().upper()
    if source_key == "NCBI":
        return NCBI_LAYOUT
    return BVBRC_LAYOUT


BVBRC_LAYOUT = build_results_layout(BVBRC_RESULTS_DIR)
NCBI_LAYOUT = build_results_layout(NCBI_RESULTS_DIR)

# Backward-compatible defaults point to the original BVBRC workflow.
FIGURES_DIR = BVBRC_LAYOUT["figures"]
RESULT_FIGURES_DIR = BVBRC_LAYOUT["result_figures"]
VALIDATION_FIGURES_DIR = BVBRC_LAYOUT["validation_figures"]
LEGACY_FIGURES_DIR = BVBRC_LAYOUT["legacy_figures"]
METRICS_DIR = BVBRC_LAYOUT["metrics"]
CURRENT_METRICS_DIR = BVBRC_LAYOUT["current_metrics"]
BENCHMARK_METRICS_DIR = BVBRC_LAYOUT["benchmark_metrics"]
ARCHIVE_METRICS_DIR = BVBRC_LAYOUT["archive_metrics"]

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
                  NCBI_FASTA_DIR, MODEL_DIR, RESULTS_DIR,
                  *BVBRC_LAYOUT.values(), *NCBI_LAYOUT.values()]:
    directory.mkdir(parents=True, exist_ok=True)

# Data source configurations
# Using BVBRC API (no authentication required - completely free!)
BVBRC_API_BASE_URL = "https://www.bv-brc.org/api"
BVBRC_AMR_PHENOTYPE_LIMIT = 50000
BVBRC_RAW_FILENAME = "ecoli_resistance_bvbrc.csv"
NCBI_RAW_FILENAME = "ecoli_resistance_ncbi.csv"
COMBINED_RAW_FILENAME = "ecoli_resistance.csv"
NCBI_ASSEMBLY_SUMMARY_REFSEQ_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_refseq.txt"
NCBI_ASSEMBLY_SUMMARY_GENBANK_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_genbank.txt"

# BVBRC provides both phenotype and genotype data
# No API keys or credentials needed! 🎉

# Antibiotics of interest (most commonly used)
COMMON_ANTIBIOTICS = [
    "CIPROFLOXACIN",
    "LEVOFLOXACIN",
    "AMOXICILLIN",
    "AMPICILLIN",
    "CEFTRIAXONE",
    "CEFOTAXIME",
    "GENTAMICIN",
    "TRIMETHOPRIM-SULFAMETHOXAZOLE",
    "TETRACYCLINE",
    "CHLORAMPHENICOL"
]

# Testing standards
TESTING_STANDARDS = ["CLSI", "EUCAST"]

# Data preprocessing parameters
MIN_SAMPLES_PER_ANTIBIOTIC = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model training parameters
CV_FOLDS = 5
N_JOBS = -1  # Use all available cores

# Hyperparameter search parameters
GRID_SEARCH_N_ITER = 50  # For randomized search
GRID_SEARCH_VERBOSE = 1

# Model validation parameters
N_PERMUTATIONS = 1000
N_MC_ITERATIONS = 100
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Plotting style
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
