# Quick Start Guide

## 🎉 NO API Keys Required!

**Great news**: This project uses BVBRC only, which requires **NO authentication**!

No Kaggle account, no API keys, no configuration needed. Just install and run! 🚀

---

## Initial Setup (2 minutes)

1. **Open PowerShell and navigate to project directory:**
   ```powershell
   cd c:\Users\cagns\Downloads\capstone_project
   ```

2. **Create and activate virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```powershell
   python -c "import sklearn, xgboost, pandas; print('Success!')"
   ```

5. **Set up Kaggle API (REQUIRED - see SETUP.md):**
   ```powershell
   # Place kaggle.json in your .kaggle folder, then test:
   kaggle datasets list
   ```

## Option 1: Run Full Pipeline (Automated)

```powershell
# Run all steps at once
python main.py --steps all --antibiotic CIPROFLOXACIN

# Or run individual steps
python main.py --steps acquire
python main.py --steps preprocess
python main.py --steps explore --antibiotic CIPROFLOXACIN
python main.py --steps train --antibiotic CIPROFLOXACIN
python main.py --steps validate
```

## Option 2: Step-by-Step (Manual Control)

### Step 1: Get Data (Week 1)

**No configuration needed - just run the script!**

```python
# Option 1: Run main pipeline
python main.py --steps acquire

# Option 2: Manual control
from src.data_acquisition import BVBRCDataDownloader

bvbrc = BVBRCDataDownloader()

# Download resistance phenotypes (replaces Kaggle dataset)
phenotypes_df = bvbrc.get_amr_phenotypes(limit=15000)
phenotypes_df.to_csv("data/raw/ecoli_resistance.csv", index=False)

# Download AMR gene data
genome_ids = phenotypes_df['genome_id'].unique()
amr_df = bvbrc.get_amr_genes(genome_ids[:500])
bvbrc.save_amr_data(amr_df)
```

**That's it!** No API keys, no credentials, just works. 🎉

### Step 2: Preprocess Data (Week 2-3)

```python
from src.data_preprocessing import EcoliDataPreprocessor

preprocessor = EcoliDataPreprocessor()

# Load data
resistance_df = preprocessor.load_resistance_data("data/raw/ecoli_resistance.csv")
genomic_df = preprocessor.load_genomic_data("data/external/amr_genes.csv")

# Clean
resistance_df = preprocessor.clean_resistance_data(resistance_df)
resistance_df = preprocessor.filter_common_antibiotics(resistance_df, min_samples=100)

# Split by standard
splits = preprocessor.split_by_testing_standard(resistance_df)

# Build features for CLSI
clsi_data = splits['CLSI']
isolates = clsi_data['isolate_id'].unique()
features = preprocessor.build_amr_feature_matrix(genomic_df, isolates)

# Merge
merged = preprocessor.merge_phenotype_genotype(clsi_data, features)
merged = preprocessor.create_binary_labels(merged)

# Save
preprocessor.save_processed_data(merged, "merged_data_CLSI.csv")
```

### Step 3: Explore Data (Week 4)

```python
from src.exploratory_analysis import ExploratoryAnalyzer

analyzer = ExploratoryAnalyzer()
df = analyzer.load_data("data/processed/merged_data_CLSI.csv")

# Summary
analyzer.explore_data_summary(df)
analyzer.plot_resistance_distribution(df)
analyzer.plot_antibiotic_comparison(df)

# PCA for specific antibiotic
df_cipro = df[df['antibiotic'] == 'CIPROFLOXACIN']
gene_cols = [c for c in df_cipro.columns if c not in 
             ['isolate_id', 'antibiotic', 'mic_value', 'resistance_phenotype',
              'testing_standard', 'binary_resistant', 'binary_resistant_liberal']]
X = df_cipro[gene_cols]
y = df_cipro['binary_resistant']

X_pca, pca = analyzer.perform_pca(X, n_components=50)
analyzer.plot_pca_variance(pca)
analyzer.plot_pca_scatter(X_pca, y)

# Clustering
k = analyzer.find_optimal_clusters(X_pca)
clusters = analyzer.perform_clustering(X_pca, n_clusters=k)
analyzer.plot_clusters(X_pca, clusters, y)
analyzer.analyze_cluster_resistance(clusters, y)
```

### Step 4: Train Models (Week 5-6)

```python
from src.model_training import train_all_models

# Prepare data
X_train, X_test, y_train, y_test = preprocessor.prepare_modeling_data(
    df, antibiotic='CIPROFLOXACIN', test_size=0.2, random_state=42
)

# Train all models (Logistic Regression, Random Forest, XGBoost)
trainer = train_all_models(
    X_train, X_test, y_train, y_test,
    antibiotic='CIPROFLOXACIN',
    cv=5
)

# Results automatically saved to results/ and models/
```

### Step 5: Validate Models (Week 7)

```python
from src.model_validation import ModelValidator

validator = ModelValidator()

# Get trained model
model = trainer.models['xgboost']
X_test_scaled = trainer.scaler.transform(X_test)

# Comprehensive validation
results = validator.comprehensive_validation(
    model,
    trainer.scaler.transform(X_train),
    X_test_scaled,
    y_train.values,
    y_test.values,
    model_name='xgboost',
    n_permutations=1000,
    n_mc_iterations=100,
    n_bootstrap=1000
)

# Check results/figures/ for plots
```

## Viewing Results

All results are saved automatically:

```powershell
# View figures
explorer results\figures

# View metrics
type results\metrics\model_comparison.csv

# Check logs
type results\pipeline.log
```

## Common Commands

```powershell
# List all antibiotics in data
python -c "import pandas as pd; df = pd.read_csv('data/processed/merged_data_CLSI.csv'); print(df['antibiotic'].value_counts())"

# Check model files
ls models

# Open Jupyter notebook for interactive analysis
jupyter notebook notebooks/

# Run specific antibiotic
python main.py --steps train validate --antibiotic LEVOFLOXACIN
```

## Troubleshooting

**Issue**: `ModuleNotFoundError`
```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Issue**: `FileNotFoundError: data/raw/ecoli_resistance.csv`
```
# Download data first:
python main.py --steps acquire
```

**Issue**: Connection error
```
# Check internet connection
# BVBRC servers might be temporarily down (rare)
# Try again in a few minutes
```

**Issue**: Out of memory
```python
# Process smaller subset:
df = df.sample(n=1000, random_state=42)  # Use 1000 samples
```

## Next Steps

1. ✅ Complete setup
2. ✅ Download data
3. ✅ Run preprocessing
4. ✅ Explore data patterns
5. ✅ Train initial models
6. ✅ Validate best model
7. 📊 Analyze results
8. 📝 Write report

## Getting Help

- Check `README.md` for full documentation
- See `GUIDE.md` for detailed analysis guide
- Review `TIMELINE.md` for project schedule
- Open `notebooks/` for interactive examples

## Project Structure at a Glance

```
capstone_project/
├── main.py              ← Run this for full pipeline
├── config.py            ← Adjust parameters here
├── requirements.txt     ← Install these packages
├── src/                 ← All analysis code
│   ├── data_acquisition.py
│   ├── data_preprocessing.py
│   ├── exploratory_analysis.py
│   ├── model_training.py
│   └── model_validation.py
├── data/                ← Your data goes here
├── models/              ← Trained models saved here
└── results/             ← Figures and metrics here
```

---

**Ready to start?** Run this:

```powershell
cd c:\Users\cagns\Downloads\capstone_project
.\venv\Scripts\Activate.ps1
python main.py --steps all --antibiotic CIPROFLOXACIN
```

Good luck with your capstone project! 🚀
