"""
Main pipeline script for E. coli Antibiotic Resistance Prediction

This script runs the complete pipeline from data acquisition to model validation.
"""

import sys
from pathlib import Path
import logging
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_acquisition import BVBRCDataDownloader
from src.data_preprocessing import EcoliDataPreprocessor
from src.exploratory_analysis import ExploratoryAnalyzer
from src.model_training import train_all_models
from src.model_validation import ModelValidator
from src.utils import setup_logging
import config

# Setup logging
setup_logging(log_file="results/pipeline.log", level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


def run_data_acquisition(skip_if_exists: bool = True):
    """
    Step 1: Download data from Kaggle and BVBRC.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA ACQUISITION")
    logger.info("="*80 + "\n")
    
    # Check if data already exists
    if skip_if_exists:
        if (config.RAW_DATA_DIR / "ecoli_resistance.csv").exists():
            logger.info("Raw data already exists. Skipping download.")
            return
    
    try:
        # Download from BVBRC (no API keys needed!)
        logger.info("Downloading from BVBRC (no credentials required)...")
        bvbrc_dl = BVBRCDataDownloader(output_dir=str(config.EXTERNAL_DATA_DIR))
        
        # Get resistance phenotypes (replaces Kaggle dataset)
        phenotypes_df = bvbrc_dl.get_amr_phenotypes(limit=15000)
        if not phenotypes_df.empty:
            phenotypes_df.to_csv(config.RAW_DATA_DIR / "ecoli_resistance.csv", index=False)
            logger.info(f"Downloaded {len(phenotypes_df)} resistance records")
        
        # Get AMR gene data - query more genomes for better model performance
        if not phenotypes_df.empty and 'genome_id' in phenotypes_df.columns:
            genome_ids = phenotypes_df['genome_id'].unique()[:2000]  # Increased from 500
            amr_df = bvbrc_dl.get_amr_genes(genome_ids)
            if not amr_df.empty:
                bvbrc_dl.save_amr_data(amr_df)
        
        logger.info("Data acquisition complete!")
        
    except Exception as e:
        logger.error(f"Error in data acquisition: {e}")
        logger.info("Check internet connection and try again.")


def run_preprocessing():
    """
    Step 2: Clean and preprocess data.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 2: DATA PREPROCESSING")
    logger.info("="*80 + "\n")
    
    preprocessor = EcoliDataPreprocessor(
        raw_data_dir=str(config.RAW_DATA_DIR),
        processed_data_dir=str(config.PROCESSED_DATA_DIR)
    )
    
    # Load resistance data
    resistance_df = preprocessor.load_resistance_data("data/raw/ecoli_resistance.csv")
    
    # Clean resistance data
    resistance_df = preprocessor.clean_resistance_data(resistance_df)
    
    # Filter for common antibiotics
    resistance_df = preprocessor.filter_common_antibiotics(
        resistance_df,
        min_samples=config.MIN_SAMPLES_PER_ANTIBIOTIC
    )
    
    # Try to load genomic data if available
    genomic_df = None
    try:
        genomic_df = preprocessor.load_genomic_data("data/external/amr_genes.csv")
        if genomic_df.empty:
            genomic_df = None
    except Exception as e:
        logger.warning(f"Could not load genomic data: {e}")
        logger.info("Will use resistance profile features instead.")
    
    # Split by testing standard
    data_splits = preprocessor.split_by_testing_standard(resistance_df)
    
    # Process each testing standard separately
    for standard, df in data_splits.items():
        logger.info(f"\nProcessing {standard} data...")
        
        if genomic_df is not None and not genomic_df.empty:
            # Use genomic features if available
            isolate_ids = df['isolate_id'].unique() if 'isolate_id' in df.columns else df['genome_id'].unique()
            feature_matrix = preprocessor.build_amr_feature_matrix(genomic_df, isolate_ids)
            merged_df = preprocessor.merge_phenotype_genotype(df, feature_matrix)
        else:
            # Use resistance data directly - will create profile features during training
            merged_df = df.copy()
            logger.info("No genomic data available - using resistance profiles as features")
        
        # Create binary labels
        merged_df = preprocessor.create_binary_labels(merged_df)
        
        # Save processed data
        preprocessor.save_processed_data(merged_df, f"merged_data_{standard}.csv")
    
    # Also save the full cleaned dataset
    preprocessor.save_processed_data(resistance_df, "cleaned_resistance_data.csv")
    
    logger.info("Preprocessing complete!")


def run_exploratory_analysis(antibiotic: str = "CIPROFLOXACIN"):
    """
    Step 3: Exploratory data analysis.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 3: EXPLORATORY ANALYSIS")
    logger.info("="*80 + "\n")
    
    analyzer = ExploratoryAnalyzer(output_dir=str(config.FIGURES_DIR))
    preprocessor = EcoliDataPreprocessor()
    
    # Try to load processed data
    try:
        df = analyzer.load_data("data/processed/merged_data_CLSI.csv")
    except FileNotFoundError:
        try:
            df = analyzer.load_data("data/processed/cleaned_resistance_data.csv")
        except FileNotFoundError:
            df = analyzer.load_data("data/raw/ecoli_resistance.csv")
            df = preprocessor.clean_resistance_data(df)
    
    # Overall exploration
    analyzer.explore_data_summary(df)
    analyzer.plot_resistance_distribution(df)
    analyzer.plot_antibiotic_comparison(df)
    
    # Antibiotic-specific analysis
    logger.info(f"\nAnalyzing {antibiotic}...")
    
    # Build resistance profile features
    X, y = preprocessor.build_resistance_profile_features(df, antibiotic)
    
    if X is not None and not X.empty and y is not None:
        # Align and clean
        common_ids = X.index.intersection(y.dropna().index)
        X = X.loc[common_ids].fillna(0)
        y = y.loc[common_ids]
        
        if len(X) > 10:
            # PCA
            n_components = min(50, X.shape[1], X.shape[0] - 1)
            X_pca, pca = analyzer.perform_pca(X, n_components=n_components)
            analyzer.plot_pca_variance(pca)
            analyzer.plot_pca_scatter(X_pca, y, f"PCA - {antibiotic}")
            
            # Clustering
            optimal_k = analyzer.find_optimal_clusters(X_pca, max_clusters=min(10, len(X)//10 + 1))
            if optimal_k > 1:
                cluster_labels = analyzer.perform_clustering(X_pca, n_clusters=optimal_k)
                analyzer.plot_clusters(X_pca, cluster_labels, y)
                analyzer.analyze_cluster_resistance(cluster_labels, y)
    else:
        logger.warning(f"Not enough data for {antibiotic} analysis")
    
    logger.info("Exploratory analysis complete!")


def run_model_training(antibiotic: str = "CIPROFLOXACIN"):
    """
    Step 4: Train machine learning models.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 4: MODEL TRAINING")
    logger.info("="*80 + "\n")
    
    # Load and prepare data
    preprocessor = EcoliDataPreprocessor()
    
    # Try to load processed data
    try:
        df = preprocessor.load_resistance_data("data/processed/cleaned_resistance_data.csv")
    except FileNotFoundError:
        df = preprocessor.load_resistance_data("data/raw/ecoli_resistance.csv")
        df = preprocessor.clean_resistance_data(df)
    
    # Use resistance profiles as features
    X_train, X_test, y_train, y_test = preprocessor.prepare_modeling_data_from_profiles(
        df,
        target_antibiotic=antibiotic,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    if X_train is None or len(X_train) == 0:
        logger.error(f"Could not prepare data for {antibiotic}")
        return None, None, None, None, None
    
    # Train all models
    trainer = train_all_models(
        X_train, X_test, y_train, y_test,
        antibiotic=antibiotic,
        cv=config.CV_FOLDS
    )
    
    logger.info("Model training complete!")
    return trainer, X_train, X_test, y_train, y_test


def run_model_validation(trainer, X_train, X_test, y_train, y_test):
    """
    Step 5: Validate models with statistical tests.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 5: MODEL VALIDATION")
    logger.info("="*80 + "\n")
    
    validator = ModelValidator(results_dir=str(config.RESULTS_DIR))
    
    # Validate each trained model
    for model_name, model in trainer.models.items():
        logger.info(f"\nValidating {model_name}...")
        
        # Prepare scaled data
        X_train_scaled = trainer.scaler.transform(X_train)
        X_test_scaled = trainer.scaler.transform(X_test)
        
        # Comprehensive validation
        results = validator.comprehensive_validation(
            model,
            X_train_scaled,
            X_test_scaled,
            y_train.values,
            y_test.values,
            model_name=model_name,
            n_permutations=config.N_PERMUTATIONS,
            n_mc_iterations=config.N_MC_ITERATIONS,
            n_bootstrap=config.N_BOOTSTRAP
        )
    
    logger.info("Model validation complete!")


def main():
    """
    Run complete pipeline.
    """
    parser = argparse.ArgumentParser(
        description="E. coli Antibiotic Resistance Prediction Pipeline"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["acquire", "preprocess", "explore", "train", "validate", "all"],
        default=["all"],
        help="Pipeline steps to run"
    )
    parser.add_argument(
        "--antibiotic",
        type=str,
        default="CIPROFLOXACIN",
        help="Antibiotic to model"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip steps if output already exists"
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("E. COLI ANTIBIOTIC RESISTANCE PREDICTION PIPELINE")
    logger.info("="*80)
    logger.info(f"Steps to run: {args.steps}")
    logger.info(f"Antibiotic: {args.antibiotic}")
    
    steps_to_run = args.steps
    if "all" in steps_to_run:
        steps_to_run = ["acquire", "preprocess", "explore", "train", "validate"]
    
    try:
        # Step 1: Data acquisition
        if "acquire" in steps_to_run:
            run_data_acquisition(skip_if_exists=args.skip_existing)
        
        # Step 2: Preprocessing
        if "preprocess" in steps_to_run:
            run_preprocessing()
        
        # Step 3: Exploratory analysis
        if "explore" in steps_to_run:
            run_exploratory_analysis(antibiotic=args.antibiotic)
        
        # Step 4: Model training
        trainer, X_train, X_test, y_train, y_test = None, None, None, None, None
        if "train" in steps_to_run:
            trainer, X_train, X_test, y_train, y_test = run_model_training(
                antibiotic=args.antibiotic
            )
        
        # Step 5: Model validation
        if "validate" in steps_to_run:
            if trainer is None:
                logger.warning("Training step was not run. Skipping validation.")
            else:
                run_model_validation(trainer, X_train, X_test, y_train, y_test)
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nResults saved to: {config.RESULTS_DIR}")
        logger.info(f"Models saved to: {config.MODEL_DIR}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
