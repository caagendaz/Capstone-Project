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

# Model directory
MODEL_DIR = PROJECT_ROOT / "models"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
                  MODEL_DIR, FIGURES_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data source configurations
# Using BVBRC API (no authentication required - completely free!)
BVBRC_API_BASE_URL = "https://www.bv-brc.org/api"

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
