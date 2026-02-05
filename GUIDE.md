# Data Analysis Guide

## Getting Started

### 1. Initial Setup

After installing dependencies, verify your setup:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import __version__ as sklearn_version
import xgboost as xgb

print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"scikit-learn: {sklearn_version}")
print(f"xgboost: {xgb.__version__}")
```

### 2. Data Acquisition

#### Option A: Using Kaggle API

```python
from src.data_acquisition import KaggleDataDownloader

# Initialize downloader
kaggle_dl = KaggleDataDownloader(output_dir="data/raw")

# Download dataset (update with actual dataset name)
kaggle_dl.download_dataset("username/ecoli-resistance-dataset")
```

#### Option B: Manual Download

1. Visit the Kaggle dataset page
2. Click "Download" button
3. Extract CSV file to `data/raw/` directory
4. Rename to `ecoli_resistance.csv`

#### BVBRC Data

```python
from src.data_acquisition import BVBRCDataDownloader

# Initialize downloader
bvbrc_dl = BVBRCDataDownloader(output_dir="data/external")

# Search for E. coli genomes
genomes_df = bvbrc_dl.search_genomes(
    organism="Escherichia coli",
    limit=1000
)

# Get AMR gene data
genome_ids = genomes_df['genome_id'].tolist()[:100]  # Start with 100
amr_df = bvbrc_dl.get_amr_genes(genome_ids)

# Save
bvbrc_dl.save_amr_data(amr_df, "amr_genes.csv")
```

### 3. Data Exploration

```python
import pandas as pd

# Load raw data
df = pd.read_csv("data/raw/ecoli_resistance.csv")

# Basic info
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Resistance distribution
print(f"\nResistance phenotypes:")
print(df['resistance_phenotype'].value_counts())

# Antibiotics tested
print(f"\nAntibiotics tested:")
print(df['antibiotic'].value_counts())
```

### 4. Data Preprocessing

```python
from src.data_preprocessing import EcoliDataPreprocessor

preprocessor = EcoliDataPreprocessor()

# Load and clean
resistance_df = preprocessor.load_resistance_data("data/raw/ecoli_resistance.csv")
resistance_df = preprocessor.clean_resistance_data(resistance_df)

# Filter common antibiotics
resistance_df = preprocessor.filter_common_antibiotics(
    resistance_df, 
    min_samples=100
)

# Check which antibiotics remain
print("Common antibiotics:")
print(resistance_df['antibiotic'].value_counts())

# Split by testing standard
data_splits = preprocessor.split_by_testing_standard(resistance_df)
print(f"\nTesting standards: {list(data_splits.keys())}")
```

### 5. Building Feature Matrix

```python
# Load genomic data
genomic_df = preprocessor.load_genomic_data("data/external/amr_genes.csv")

# Build feature matrix for CLSI data
clsi_data = data_splits['CLSI']
isolate_ids = clsi_data['isolate_id'].unique()

feature_matrix = preprocessor.build_amr_feature_matrix(
    genomic_df, 
    isolate_ids
)

print(f"Feature matrix shape: {feature_matrix.shape}")
print(f"Number of AMR genes: {feature_matrix.shape[1]}")
print(f"\nFirst few genes: {feature_matrix.columns[:10].tolist()}")

# Merge with phenotypes
merged_df = preprocessor.merge_phenotype_genotype(clsi_data, feature_matrix)
merged_df = preprocessor.create_binary_labels(merged_df)

# Save
preprocessor.save_processed_data(merged_df, "merged_data_CLSI.csv")
```

### 6. Exploratory Analysis

```python
from src.exploratory_analysis import ExploratoryAnalyzer

analyzer = ExploratoryAnalyzer()

# Load processed data
df = analyzer.load_data("data/processed/merged_data_CLSI.csv")

# Summary statistics
analyzer.explore_data_summary(df)

# Visualizations
analyzer.plot_resistance_distribution(df)
analyzer.plot_antibiotic_comparison(df)
```

### 7. PCA and Clustering

```python
# Select specific antibiotic
antibiotic = "CIPROFLOXACIN"
df_abx = df[df['antibiotic'] == antibiotic]

# Separate features (AMR genes) from metadata
metadata_cols = ['isolate_id', 'antibiotic', 'mic_value',
                'resistance_phenotype', 'testing_standard',
                'binary_resistant', 'binary_resistant_liberal']
gene_cols = [col for col in df_abx.columns if col not in metadata_cols]

X = df_abx[gene_cols]
y = df_abx['binary_resistant']

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# PCA
X_pca, pca = analyzer.perform_pca(X, n_components=50)
analyzer.plot_pca_variance(pca)
analyzer.plot_pca_scatter(X_pca, y, f"PCA - {antibiotic}")

# Clustering
optimal_k = analyzer.find_optimal_clusters(X_pca, max_clusters=10)
cluster_labels = analyzer.perform_clustering(X_pca, n_clusters=optimal_k)
analyzer.plot_clusters(X_pca, cluster_labels, y)
analyzer.analyze_cluster_resistance(cluster_labels, y)
```

### 8. Model Training

```python
from src.model_training import train_all_models

# Prepare data
X_train, X_test, y_train, y_test = preprocessor.prepare_modeling_data(
    df,
    antibiotic="CIPROFLOXACIN",
    test_size=0.2,
    random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Class distribution (train): {y_train.value_counts().to_dict()}")

# Train all models
trainer = train_all_models(
    X_train, X_test, y_train, y_test,
    antibiotic="CIPROFLOXACIN",
    cv=5
)

# Compare models
comparison = trainer.compare_models()
print("\nModel Comparison:")
print(comparison)
```

### 9. Model Validation

```python
from src.model_validation import ModelValidator

validator = ModelValidator()

# Get best model (e.g., XGBoost)
model = trainer.models['xgboost']
X_train_scaled = trainer.scaler.transform(X_train)
X_test_scaled = trainer.scaler.transform(X_test)

# Permutation test
perm_results, perm_scores = validator.permutation_test(
    model, X_test_scaled, y_test.values,
    n_permutations=1000,
    metric='accuracy'
)
print(f"\nPermutation test p-value: {perm_results['p_value']:.4f}")

# Monte Carlo simulation
X_all = np.vstack([X_train_scaled, X_test_scaled])
y_all = np.concatenate([y_train.values, y_test.values])

mc_results = validator.monte_carlo_simulation(
    model, X_all, y_all,
    n_iterations=100
)

# Bootstrap confidence intervals
ci_results = {}
for metric in ['accuracy', 'f1', 'roc_auc']:
    mean, lower, upper = validator.bootstrap_confidence_interval(
        model, X_test_scaled, y_test.values,
        metric=metric,
        n_bootstrap=1000
    )
    ci_results[metric] = (mean, lower, upper)
    print(f"\n{metric}: {mean:.4f} [{lower:.4f}, {upper:.4f}]")
```

### 10. Making Predictions

```python
import joblib

# Load saved model
model = joblib.load("models/xgboost_CIPROFLOXACIN_20260205_120000.pkl")
scaler = joblib.load("models/scaler_CIPROFLOXACIN_20260205_120000.pkl")

# Prepare new data (example)
# new_isolate = df_abx[gene_cols].iloc[0:1]  # One isolate
# new_isolate_scaled = scaler.transform(new_isolate)

# Predict
# prediction = model.predict(new_isolate_scaled)
# probability = model.predict_proba(new_isolate_scaled)[:, 1]

# print(f"Prediction: {'Resistant' if prediction[0] == 1 else 'Susceptible'}")
# print(f"Probability of resistance: {probability[0]:.2%}")
```

## Common Issues and Solutions

### Issue 1: Kaggle API Not Configured

**Error**: `OSError: Could not find kaggle.json`

**Solution**:
1. Create Kaggle account
2. Go to Account → API → Create New API Token
3. Move downloaded `kaggle.json` to `~\.kaggle\` on Windows
4. Run: `kaggle datasets list` to verify

### Issue 2: BVBRC API Rate Limiting

**Error**: Too many requests

**Solution**:
- Add delays between requests (already implemented)
- Reduce batch size when downloading AMR data
- Use cached data when available

### Issue 3: Memory Issues with Large Datasets

**Solution**:
- Process data in chunks
- Use `pd.read_csv(..., chunksize=10000)`
- Filter to specific antibiotics early
- Use sparse matrices for AMR genes

### Issue 4: Imbalanced Classes

**Symptoms**: Poor precision or recall

**Solution**:
- Use `class_weight='balanced'` in models
- Consider SMOTE for oversampling
- Focus on ROC-AUC instead of accuracy
- Adjust decision threshold

### Issue 5: Long Training Times

**Solution**:
- Reduce hyperparameter search space
- Use fewer CV folds (3 instead of 5)
- Reduce `n_iter` in RandomizedSearchCV
- Use `n_jobs=-1` for parallelization

## Next Steps

1. **Try different antibiotics**: Repeat analysis for other antibiotics
2. **Feature selection**: Remove low-importance genes to simplify models
3. **Ensemble methods**: Combine predictions from multiple models
4. **Temporal validation**: Split data by time if available
5. **External validation**: Test on independent dataset
6. **Clinical interpretation**: Consult with clinicians on results
7. **Web interface**: Build simple prediction tool

## Additional Resources

- **Kaggle**: https://www.kaggle.com/
- **BVBRC**: https://www.bv-brc.org/
- **Scikit-learn docs**: https://scikit-learn.org/
- **XGBoost docs**: https://xgboost.readthedocs.io/
- **AMR resources**: https://www.cdc.gov/drugresistance/
