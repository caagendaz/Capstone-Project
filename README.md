# Predicting Antibacterial Resistance in E. coli Using Genomic Features

A comprehensive machine learning pipeline for predicting antibiotic resistance in *Escherichia coli* based on genomic antimicrobial resistance (AMR) gene profiles.

## Project Overview

Antibiotic-resistant bacteria represent one of the most significant public health challenges. This project develops machine learning models to predict antibiotic resistance phenotypes in E. coli using genomic features, enabling healthcare professionals to tailor antibiotic treatments based on bacterial strain characteristics.

### Key Features

- **Multi-source data integration**: Combines phenotypic resistance data with genomic AMR gene profiles
- **Multiple ML algorithms**: Logistic Regression, Random Forest, and XGBoost
- **Rigorous validation**: Includes permutation tests, Monte Carlo simulations, and bootstrap confidence intervals
- **Exploratory analysis**: PCA and K-means clustering for pattern discovery
- **Production-ready**: Modular, well-documented code with comprehensive error handling

## Project Structure

```
capstone_project/
├── data/
│   ├── raw/                    # Raw data from Kaggle
│   ├── processed/              # Cleaned and merged datasets
│   └── external/               # BVBRC genomic data
├── models/                     # Trained models and scalers
├── results/
│   ├── figures/                # Visualizations
│   └── metrics/                # Performance metrics
├── src/
│   ├── data_acquisition.py     # Data download scripts
│   ├── data_preprocessing.py   # Data cleaning and feature engineering
│   ├── exploratory_analysis.py # PCA and clustering
│   ├── model_training.py       # Model training and hyperparameter tuning
│   ├── model_validation.py     # Statistical validation
│   └── utils.py                # Helper functions
├── notebooks/                  # Jupyter notebooks for analysis
├── tests/                      # Unit tests
├── config.py                   # Configuration parameters
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for downloading data from BVBRC)

### 🎉 NO API Keys Required!

**Good news**: This project uses BVBRC (Bacterial and Viral Bioinformatics Resource Center) which requires **NO authentication**! No Kaggle account, no API keys, nothing to configure.

See **[NO_API_KEYS.md](NO_API_KEYS.md)** for details.

### Quick Installation

1. Navigate to the project directory:
```powershell
cd c:\Users\cagns\Downloads\capstone_project
```

2. Create a virtual environment (recommended):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install required packages:
```powershell
pip install -r requirements.txt
```

4. Verify setup (optional):
```powershell
python check_setup.py
```

5. Run the pipeline:
```powershell
python main.py --steps all --antibiotic CIPROFLOXACIN
```

The code will automatically download all data from BVBRC (no credentials needed)!

## Data Sources

### BVBRC (Bacterial and Viral Bioinformatics Resource Center)

**Everything comes from BVBRC - NO API keys required!** 🎉

- **Description**: Comprehensive bacterial genomics resource
- **What we get**: 
  - Resistance phenotypes (MIC values, S/I/R calls)
  - AMR gene annotations
  - Genome metadata
- **Size**: ~15,000 E. coli isolates with resistance data
- **Features**: 
  - MIC values for various antibiotics
  - Resistance phenotypes (S/I/R)
  - Testing standards (CLSI/EUCAST)
  - AMR gene presence/absence
- **API**: https://www.bv-brc.org/api/
- **Access**: Completely free and open (no authentication)
- **Documentation**: https://www.bv-brc.org/docs/

The code automatically downloads everything you need!

## Usage

### Quick Start

Run the complete pipeline:

```python
# 1. Download data
python src/data_acquisition.py

# 2. Preprocess data
python src/data_preprocessing.py

# 3. Exploratory analysis
python src/exploratory_analysis.py

#### 1. Data Acquisition

```python
from src.data_acquisition import BVBRCDataDownloader

# Download from BVBRC (no credentials needed!)
bvbrc = BVBRCDataDownloader()

# Get resistance phenotypes (automatically downloads to data/raw/)
phenotypes_df = bvbrc.get_amr_phenotypes(limit=15000)

# Get AMR gene data
genome_ids = phenotypes_df['genome_id'].unique()
amr_df = bvbrc.get_amr_genes(genome_ids)
```
# Download from Kaggle
kaggle_dl = KaggleDataDownloader()
kaggle_dl.download_dataset("username/ecoli-resistance-dataset")

# Download from BVBRC
bvbrc_dl = BVBRCDataDownloader()
genomes_df = bvbrc_dl.search_genomes(organism="Escherichia coli", limit=1000)
amr_df = bvbrc_dl.get_amr_genes(genomes_df['genome_id'].tolist())
```

#### 2. Data Preprocessing

```python
from src.data_preprocessing import EcoliDataPreprocessor

preprocessor = EcoliDataPreprocessor()

# Load and clean data
resistance_df = preprocessor.load_resistance_data("data/raw/ecoli_resistance.csv")
resistance_df = preprocessor.clean_resistance_data(resistance_df)

# Filter for common antibiotics
resistance_df = preprocessor.filter_common_antibiotics(resistance_df, min_samples=100)

# Split by testing standard
data_splits = preprocessor.split_by_testing_standard(resistance_df)

# Build AMR gene feature matrix
genomic_df = preprocessor.load_genomic_data("data/external/amr_genes.csv")
feature_matrix = preprocessor.build_amr_feature_matrix(
    genomic_df, 
    resistance_df['isolate_id'].unique()
)

# Merge phenotype and genotype
merged_df = preprocessor.merge_phenotype_genotype(resistance_df, feature_matrix)
merged_df = preprocessor.create_binary_labels(merged_df)

# Save processed data
preprocessor.save_processed_data(merged_df, "merged_data_CLSI.csv")
```

#### 3. Exploratory Analysis

```python
from src.exploratory_analysis import ExploratoryAnalyzer

analyzer = ExploratoryAnalyzer()

# Load processed data
df = analyzer.load_data("data/processed/merged_data_CLSI.csv")

# Basic exploration
analyzer.explore_data_summary(df)
analyzer.plot_resistance_distribution(df)
analyzer.plot_antibiotic_comparison(df)

# PCA for dimensionality reduction
X = df[gene_columns]  # Select AMR gene columns
y = df['binary_resistant']

X_pca, pca = analyzer.perform_pca(X, n_components=50)
analyzer.plot_pca_variance(pca)
analyzer.plot_pca_scatter(X_pca, y)

# K-means clustering
optimal_k = analyzer.find_optimal_clusters(X_pca, max_clusters=10)
cluster_labels = analyzer.perform_clustering(X_pca, n_clusters=optimal_k)
analyzer.plot_clusters(X_pca, cluster_labels, y)
analyzer.analyze_cluster_resistance(cluster_labels, y)
```

#### 4. Model Training

```python
from src.model_training import train_all_models
from src.data_preprocessing import EcoliDataPreprocessor

preprocessor = EcoliDataPreprocessor()
df = preprocessor.load_data("data/processed/merged_data_CLSI.csv")

# Prepare data for specific antibiotic
antibiotic = "CIPROFLOXACIN"
X_train, X_test, y_train, y_test = preprocessor.prepare_modeling_data(
    df, antibiotic, test_size=0.2, random_state=42
)

# Train all models (Logistic Regression, Random Forest, XGBoost)
trainer = train_all_models(
    X_train, X_test, y_train, y_test,
    antibiotic=antibiotic,
    cv=5
)

# Models are automatically saved to models/ directory
# Results saved to results/ directory
```

#### 5. Model Validation

```python
from src.model_validation import ModelValidator
import joblib

# Load trained model
model = joblib.load("models/xgboost_CIPROFLOXACIN_20260205_120000.pkl")

validator = ModelValidator()

# Comprehensive validation
results = validator.comprehensive_validation(
    model, X_train, X_test, y_train, y_test,
    model_name='xgboost',
    n_permutations=1000,
    n_mc_iterations=100,
    n_bootstrap=1000
)

# Individual tests
# Permutation test
perm_results, perm_scores = validator.permutation_test(
    model, X_test, y_test, n_permutations=1000
)

# Monte Carlo simulation
mc_results = validator.monte_carlo_simulation(
    model, X_all, y_all, n_iterations=100
)

# Bootstrap confidence intervals
ci_results = {}
for metric in ['accuracy', 'f1', 'roc_auc']:
    mean, lower, upper = validator.bootstrap_confidence_interval(
        model, X_test, y_test, metric=metric, n_bootstrap=1000
    )
    ci_results[metric] = (mean, lower, upper)
```

## Methodology

### 1. Data Processing
- **Missing value handling**: Rows with missing phenotypes removed; MIC values handled case-by-case
- **Testing standard normalization**: Separate models for CLSI vs EUCAST standards
- **Binary classification**: Convert S/I/R to binary (Resistant vs Non-resistant)
- **Feature engineering**: AMR gene presence/absence binary matrix

### 2. Exploratory Analysis
- **PCA**: Reduce dimensionality while retaining 95% variance
- **K-means clustering**: Identify natural groupings in genomic profiles
- **Visualization**: Scatter plots, heatmaps, resistance rate comparisons

### 3. Model Training
Three algorithms selected for complementary strengths:

- **Logistic Regression**: Linear baseline, highly interpretable
- **Random Forest**: Handles non-linear interactions, provides feature importance
- **XGBoost**: State-of-the-art gradient boosting, excellent for high-dimensional genomic data

**Hyperparameter optimization**:
- Grid search for Logistic Regression
- Randomized search for Random Forest and XGBoost (50 iterations)
- 5-fold stratified cross-validation
- ROC-AUC as primary optimization metric

### 4. Model Validation

**Permutation Tests** (n=1000):
- Null hypothesis: Model predictions independent of true labels
- Tests statistical significance of model performance
- P-value < 0.05 indicates significant performance

**Monte Carlo Simulations** (n=100):
- Repeated random train/test splits
- Assesses model stability across different data samples
- Quantifies variance due to sampling

**Bootstrap Confidence Intervals** (n=1000):
- 95% CI for accuracy, precision, recall, F1, ROC-AUC
- Non-parametric estimation of metric uncertainty
- Enables robust performance reporting

## Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Confusion Matrix**: Detailed breakdown of predictions

## Results Interpretation

### Model Comparison
Compare models based on:
1. ROC-AUC (primary metric for imbalanced classes)
2. F1 score (balance precision and recall)
3. Stability (narrow confidence intervals)
4. Statistical significance (permutation test p-values)

### Feature Importance
- Tree-based models provide gene-level importance scores
- Identifies key AMR genes driving resistance predictions
- Enables biological interpretation of model decisions

### Clinical Utility
- High precision: Avoid unnecessary broad-spectrum antibiotics
- High recall: Ensure resistant strains are caught
- Trade-off depends on clinical context and cost of errors

## Timeline

| Week | Task |
|------|------|
| 1 | Download datasets and extract relevant variables |
| 2 | Build data merging pipeline and clean dataset |
| 3 | Build gene presence/absence matrix, split by testing standard |
| 4 | PCA and clustering analysis |
| 5 | Train predictive models |
| 6 | Hyperparameter optimization and evaluation |
| 7 | Permutation tests, Monte Carlo, confidence intervals |
| 8-10 | Interpret results and complete writeup |

## Expected Contributions

### Scientific Impact
- Bridge gap between AMR gene identification and resistance prediction
- Provide framework for genomic epidemiology of resistance
- Enable data-driven antibiotic stewardship

### Practical Applications
- **Clinical**: Rapid resistance prediction for treatment selection
- **Public Health**: Surveillance of resistance patterns
- **Veterinary**: Monitor farm animal resistance
- **Research**: Identify novel resistance mechanisms

## Limitations and Future Work

### Current Limitations
- Dataset limited to E. coli (not generalizable to other species)
- Different testing standards may affect generalizability
- Requires genomic sequencing (not always available clinically)
- Binary classification simplifies complex resistance mechanisms

### Future Directions
- Extend to multi-species resistance prediction
- Incorporate environmental and epidemiological features
- Develop online prediction tool with web interface
- Real-time model updating with new resistance data
- Integration with clinical decision support systems

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- Python 3.8+
- scikit-learn 1.0+
- XGBoost 1.5+
- pandas, numpy, scipy
- matplotlib, seaborn
- tqdm (progress bars)
- kaggle (data download)

## Contributing

This is a capstone project. For questions or collaboration:
- Open an issue in the repository
- Contact: [Your email]

## License

This project is for educational purposes. Data sources have their own licenses:
- Kaggle datasets: Check individual dataset licenses
- BVBRC data: Public domain

## Acknowledgments

- **Kaggle community**: For curated resistance datasets
- **BVBRC**: For comprehensive bacterial genomics resources
- **Scikit-learn developers**: For robust ML tools
- **Academic advisors**: For project guidance

## Citation

If you use this code or methodology, please cite:

```
[Your Name] (2026). Predicting Antibacterial Resistance in E. coli Using Genomic Features.
Capstone Project. [Your Institution].
```

## Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **Institution**: Your Institution
- **Date**: February 2026

---

**Note**: This README assumes you will update placeholder values (Kaggle dataset names, author information, etc.) with actual project-specific details.
