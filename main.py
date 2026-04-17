"""
Main pipeline script for E. coli Antibiotic Resistance Prediction

This script runs the complete pipeline from data acquisition to model validation.
"""

import sys
from pathlib import Path
import logging
import argparse
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_acquisition import BVBRCDataDownloader, NCBIASTDataImporter
from src.data_preprocessing import EcoliDataPreprocessor
from src.exploratory_analysis import ExploratoryAnalyzer
from src.model_training import train_all_models
from src.model_validation import ModelValidator
import config

logger = logging.getLogger(__name__)


def configure_pipeline_logging(results_root: Path):
    """Attach a file logger for the selected results root."""
    log_path = results_root / "pipeline.log"
    formatter = logging.Formatter(config.LOG_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    has_stream_handler = any(isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers)
    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def resolve_results_source(feature_source: str) -> str:
    """Map feature sources to their default results namespace."""
    return "NCBI" if feature_source.lower() == "ncbi_genes" else "BVBRC"


def run_data_acquisition(skip_if_exists: bool = True,
                         acquisition_source: str = "bvbrc",
                         ncbi_ast_file: str | None = None,
                         combine_sources: bool = False):
    """
    Step 1: Download or import phenotype/genotype data.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA ACQUISITION")
    logger.info("="*80 + "\n")

    acquisition_source = acquisition_source.lower()

    # Check if canonical raw data already exists
    if skip_if_exists and (config.RAW_DATA_DIR / config.COMBINED_RAW_FILENAME).exists():
        logger.info("Canonical raw data already exists. Skipping acquisition.")
        return

    try:
        phenotype_frames = []

        if acquisition_source in {"bvbrc", "both"}:
            logger.info("Downloading from BVBRC (no credentials required)...")
            bvbrc_dl = BVBRCDataDownloader(output_dir=str(config.EXTERNAL_DATA_DIR))

            phenotypes_df = bvbrc_dl.get_amr_phenotypes(limit=config.BVBRC_AMR_PHENOTYPE_LIMIT)
            if not phenotypes_df.empty:
                phenotypes_df = phenotypes_df.copy()
                phenotypes_df["source_database"] = "BVBRC"
                phenotypes_df.to_csv(config.RAW_DATA_DIR / config.BVBRC_RAW_FILENAME, index=False)
                phenotype_frames.append(("bvbrc", phenotypes_df))
                logger.info(f"Downloaded {len(phenotypes_df)} BVBRC resistance records")

                # Get AMR gene data for ALL phenotype genomes (no limit)
                if 'genome_id' in phenotypes_df.columns:
                    genome_ids = phenotypes_df['genome_id'].unique()
                    logger.info(f"Querying AMR genes for {len(genome_ids)} genomes...")
                    amr_df = bvbrc_dl.get_amr_genes(genome_ids)
                    if not amr_df.empty:
                        bvbrc_dl.save_amr_data(amr_df)

        if acquisition_source in {"ncbi", "both"}:
            if not ncbi_ast_file:
                raise ValueError(
                    "NCBI acquisition requires --ncbi-ast-file pointing to an AST Browser CSV/TSV export"
                )

            logger.info("Importing NCBI AST Browser export from %s...", ncbi_ast_file)
            ncbi_importer = NCBIASTDataImporter(output_dir=str(config.RAW_DATA_DIR))
            ncbi_df = ncbi_importer.import_ast_export(ncbi_ast_file)
            ncbi_importer.save_phenotype_data(ncbi_df, config.NCBI_RAW_FILENAME)
            phenotype_frames.append(("ncbi", ncbi_df))

        if not phenotype_frames:
            logger.warning("No phenotype data was acquired")
            return

        if len(phenotype_frames) == 1:
            selected_name, selected_df = phenotype_frames[0]
            selected_df.to_csv(config.RAW_DATA_DIR / config.COMBINED_RAW_FILENAME, index=False)
            logger.info(
                "Saved %s data as canonical phenotype file: %s",
                selected_name.upper(),
                config.RAW_DATA_DIR / config.COMBINED_RAW_FILENAME
            )
        elif combine_sources:
            combined_df = pd.concat([df for _, df in phenotype_frames], ignore_index=True, sort=False)
            before = len(combined_df)
            combined_df = combined_df.drop_duplicates()
            combined_df.to_csv(config.RAW_DATA_DIR / config.COMBINED_RAW_FILENAME, index=False)
            logger.warning(
                "Combined %s phenotype rows into %s after exact deduplication only. "
                "Cross-repository duplicate isolate detection is not guaranteed.",
                before,
                len(combined_df)
            )
        else:
            logger.info(
                "Saved source-specific phenotype files only. Canonical %s was left unchanged because "
                "--combine-sources was not enabled.",
                config.COMBINED_RAW_FILENAME
            )

        logger.info("Data acquisition complete!")

    except Exception as e:
        logger.error(f"Error in data acquisition: {e}")
        logger.info("Check internet connection / file path and try again.")


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
    
    Uses AMR gene presence/absence features (not antibiotic resistance profiles)
    for clustering and PCA analysis.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 3: EXPLORATORY ANALYSIS")
    logger.info("="*80 + "\n")
    
    analyzer = ExploratoryAnalyzer(output_dir=str(config.RESULT_FIGURES_DIR))
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
    
    # Antibiotic-specific analysis using AMR GENE features
    logger.info(f"\nAnalyzing {antibiotic} using AMR gene features...")
    
    # Identify gene feature columns (exclude metadata columns)
    metadata_cols = ['measurement', 'date_inserted', 'date_modified', 'laboratory_typing_method',
                    'laboratory_typing_platform', 'laboratory_typing_method_version',
                    'taxon_id', 'measurement_value', 'evidence', 'resistant_phenotype',
                    'resistance_phenotype', 'public', 'vendor', 'id', 'measurement_sign',
                    'antibiotic', 'testing_standard', 'testing_standard_year', 'owner',
                    'pmid', 'genome_id', 'genome_name', '_version_', 'measurement_unit',
                    'source', 'mic_value', 'isolate_id', 'binary_resistant']
    
    gene_cols = [c for c in df.columns if c not in metadata_cols]
    
    if len(gene_cols) > 5:
        logger.info(f"Found {len(gene_cols)} AMR gene features")
        logger.info(f"Sample genes: {gene_cols[:10]}")
        
        # Extract features and labels
        X = df[gene_cols].fillna(0)
        
        # Get target label
        if 'binary_resistant' in df.columns:
            y = df['binary_resistant']
        elif 'resistant_phenotype' in df.columns:
            y = (df['resistant_phenotype'] == 'RESISTANT').astype(int)
        elif 'resistance_phenotype' in df.columns:
            y = (df['resistance_phenotype'] == 'R').astype(int)
        else:
            logger.warning("No resistance label found")
            y = None
        
        logger.info(f"Feature matrix shape: {X.shape} (samples x genes)")
        
        if len(X) > 10:
            # PCA on gene features
            n_components = min(50, X.shape[1], X.shape[0] - 1)
            X_pca, pca = analyzer.perform_pca(X, n_components=n_components)
            analyzer.plot_pca_variance(pca)
            if y is not None:
                analyzer.plot_pca_scatter(X_pca, y, f"PCA - {antibiotic} (AMR Genes)")
            
            # Clustering - need enough samples for meaningful elbow plot
            min_k = 6  # Minimum range for elbow: k=[2,3,4,5,6]
            max_k = min(10, max(min_k, len(X)//10 + 1))
            optimal_k = analyzer.find_optimal_clusters(X_pca, max_clusters=max_k)
            if optimal_k > 1:
                cluster_labels = analyzer.perform_clustering(X_pca, n_clusters=optimal_k)
                if y is not None:
                    analyzer.plot_clusters(X_pca, cluster_labels, y)
                    analyzer.analyze_cluster_resistance(cluster_labels, y)
        else:
            logger.warning(f"Only {len(X)} samples - need more data for PCA/clustering")
    else:
        # Fallback to resistance profile features if no gene data
        logger.info("No AMR gene features found, falling back to resistance profiles...")
        X, y = preprocessor.build_resistance_profile_features(df, antibiotic)
        
        if X is not None and not X.empty and y is not None:
            common_ids = X.index.intersection(y.dropna().index)
            X = X.loc[common_ids].fillna(0)
            y = y.loc[common_ids]
            
            if len(X) > 10:
                n_components = min(50, X.shape[1], X.shape[0] - 1)
                X_pca, pca = analyzer.perform_pca(X, n_components=n_components)
                analyzer.plot_pca_variance(pca)
                analyzer.plot_pca_scatter(X_pca, y, f"PCA - {antibiotic}")
                
                min_k = 6
                max_k = min(10, max(min_k, len(X)//10 + 1))
                optimal_k = analyzer.find_optimal_clusters(X_pca, max_clusters=max_k)
                if optimal_k > 1:
                    cluster_labels = analyzer.perform_clustering(X_pca, n_clusters=optimal_k)
                    analyzer.plot_clusters(X_pca, cluster_labels, y)
                    analyzer.analyze_cluster_resistance(cluster_labels, y)
    
    logger.info("Exploratory analysis complete!")


def run_model_training(antibiotic: str = "CIPROFLOXACIN"):
    """
    Step 4: Train machine learning models.
    """
    return run_model_training_with_source(
        antibiotic=antibiotic,
        feature_source="profiles",
        testing_standard="EUCAST",
        cv=config.CV_FOLDS,
        use_ensemble=True
    )


def prepare_training_data(antibiotic: str,
                          feature_source: str = "profiles",
                          testing_standard: str = "EUCAST",
                          split_group: str = "sample"):
    """
    Load and prepare training data for a selected feature source.
    """
    preprocessor = EcoliDataPreprocessor()
    feature_source = feature_source.lower()
    testing_standard = testing_standard.upper()

    if feature_source == "profiles":
        try:
            df = preprocessor.load_resistance_data("data/processed/cleaned_resistance_data.csv")
        except FileNotFoundError:
            df = preprocessor.load_resistance_data("data/raw/ecoli_resistance.csv")
            df = preprocessor.clean_resistance_data(df)

        return preprocessor.prepare_modeling_data_from_profiles(
            df,
            target_antibiotic=antibiotic,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )

    if feature_source == "genes":
        include_testing_standard_feature = False

        if testing_standard == "COMBINED":
            eucast_df = preprocessor.load_resistance_data("data/processed/merged_data_EUCAST.csv")
            clsi_df = preprocessor.load_resistance_data("data/processed/merged_data_CLSI.csv")

            all_cols = sorted(set(eucast_df.columns) | set(clsi_df.columns))
            df = pd.concat(
                [eucast_df.reindex(columns=all_cols), clsi_df.reindex(columns=all_cols)],
                ignore_index=True
            )

            id_col = 'isolate_id' if 'isolate_id' in df.columns else 'genome_id'
            if id_col in df.columns and 'testing_standard' in df.columns:
                df['_standard_priority'] = (
                    df['testing_standard'].astype(str).str.upper().map({'EUCAST': 0, 'CLSI': 1}).fillna(2)
                )
                df = (
                    df.sort_values([id_col, '_standard_priority'], kind='mergesort')
                      .drop_duplicates(subset=[id_col, 'antibiotic'], keep='first')
                      .drop(columns=['_standard_priority'])
                )

            include_testing_standard_feature = True
        else:
            merged_path = f"data/processed/merged_data_{testing_standard}.csv"
            df = preprocessor.load_resistance_data(merged_path)

        return preprocessor.prepare_modeling_data(
            df,
            antibiotic=antibiotic,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            include_testing_standard_feature=include_testing_standard_feature
        )

    if feature_source == "ncbi_genes":
        quinolone_matrix_path = Path(
            "data/external/amrfinder_batch_cipro_remaining/amrfinder_quinolone_labeled_feature_matrix.csv"
        )
        completed_matrix_path = Path(
            "data/external/amrfinder_batch_cipro_remaining/amrfinder_labeled_feature_matrix.csv"
        )
        pilot_matrix_path = Path(
            "data/external/amrfinder_batch_cipro_pilot25/amrfinder_labeled_feature_matrix.csv"
        )

        if quinolone_matrix_path.exists():
            matrix_path = quinolone_matrix_path
        elif completed_matrix_path.exists():
            matrix_path = completed_matrix_path
        else:
            matrix_path = pilot_matrix_path
        labeled_matrix = preprocessor.load_resistance_data(str(matrix_path))

        if (
            split_group == "bioproject_accession"
            and (
                "bioproject_accession" not in labeled_matrix.columns
                or labeled_matrix["bioproject_accession"].isna().all()
            )
            and "biosample_accession" in labeled_matrix.columns
        ):
            bioproject_map_path = Path("data/external/biosample_to_assembly.csv")
            if bioproject_map_path.exists():
                bioproject_map = (
                    pd.read_csv(
                        bioproject_map_path,
                        usecols=["biosample_accession", "bioproject_accession"],
                        low_memory=False
                    )
                    .drop_duplicates(subset=["biosample_accession"])
                )
                labeled_matrix = labeled_matrix.merge(
                    bioproject_map,
                    on="biosample_accession",
                    how="left"
                )

        label_col = (
            "resistance_phenotype"
            if "resistance_phenotype" in labeled_matrix.columns
            else "ciprofloxacin_phenotype"
        )
        return preprocessor.prepare_modeling_data_from_feature_matrix(
            labeled_matrix,
            antibiotic=antibiotic,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            feature_prefix="feat_",
            label_col=label_col,
            positive_label="R",
            sample_id_col="Name",
            split_group_col=split_group
        )

    raise ValueError(f"Unknown feature source: {feature_source}")


def run_model_training_with_source(antibiotic: str = "CIPROFLOXACIN",
                                   feature_source: str = "profiles",
                                   testing_standard: str = "EUCAST",
                                   split_group: str = "sample",
                                   cv: int = 5,
                                   use_ensemble: bool = True,
                                   metrics_dir: str | None = None,
                                   results_source: str | None = None):
    """
    Step 4: Train machine learning models for a selected feature source.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 4: MODEL TRAINING")
    logger.info("="*80 + "\n")

    logger.info(f"Feature source: {feature_source}")
    if feature_source.lower() == "genes":
        logger.info(f"Testing standard: {testing_standard}")
    if feature_source.lower() == "ncbi_genes" and split_group != "sample":
        logger.info(f"Strict split group: {split_group}")

    selected_results_source = (results_source or resolve_results_source(feature_source)).upper()
    results_layout = config.get_results_layout(selected_results_source)
    logger.info(f"Results source: {selected_results_source}")

    X_train, X_test, y_train, y_test = prepare_training_data(
        antibiotic=antibiotic,
        feature_source=feature_source,
        testing_standard=testing_standard,
        split_group=split_group
    )
    
    if X_train is None or len(X_train) == 0:
        logger.error(f"Could not prepare data for {antibiotic} using {feature_source}")
        return None, None, None, None, None
    
    # Train all models
    split_suffix = ""
    if feature_source.lower() == "ncbi_genes" and split_group != "sample":
        split_suffix = f"_{split_group.upper()}"

    trainer = train_all_models(
        X_train, X_test, y_train, y_test,
        antibiotic=(
            f"{antibiotic}_{feature_source.upper()}{split_suffix}"
            if feature_source.lower() in {"profiles", "ncbi_genes"}
            else f"{antibiotic}_{feature_source.upper()}_{testing_standard}"
        ),
        results_dir=str(results_layout["root"]),
        figures_dir=str(results_layout["result_figures"]),
        metrics_dir=metrics_dir or str(results_layout["current_metrics"]),
        cv=cv,
        use_ensemble=use_ensemble
    )
    
    logger.info("Model training complete!")
    return trainer, X_train, X_test, y_train, y_test


def run_model_validation(trainer, X_train, X_test, y_train, y_test, results_source: str = "BVBRC"):
    """
    Step 5: Validate models with statistical tests.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 5: MODEL VALIDATION")
    logger.info("="*80 + "\n")
    
    results_layout = config.get_results_layout(results_source)
    validator = ModelValidator(
        results_dir=str(results_layout["root"]),
        figures_dir=str(results_layout["validation_figures"])
    )
    
    # Validate each trained model
    for model_name, model in trainer.models.items():
        logger.info(f"\nValidating {model_name}...")

        X_train_eval = X_train.values if hasattr(X_train, 'values') else X_train
        X_test_eval = X_test.values if hasattr(X_test, 'values') else X_test

        # Comprehensive validation
        results = validator.comprehensive_validation(
            model,
            X_train_eval,
            X_test_eval,
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
        "--feature-source",
        choices=["profiles", "genes", "ncbi_genes"],
        default="profiles",
        help="Feature source to train on"
    )
    parser.add_argument(
        "--testing-standard",
        choices=["CLSI", "EUCAST", "COMBINED"],
        default="EUCAST",
        help="Testing standard to use for gene-based training"
    )
    parser.add_argument(
        "--results-source",
        choices=["BVBRC", "NCBI"],
        default=None,
        help="Results namespace to write into; defaults based on feature source"
    )
    parser.add_argument(
        "--split-group",
        default="sample",
        help=(
            "Grouping variable for stricter holdout splits. For ncbi_genes, "
            "use 'sample' (default), 'bioproject_accession', or 'location_source'."
        )
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip steps if output already exists"
    )
    parser.add_argument(
        "--acquisition-source",
        choices=["bvbrc", "ncbi", "both"],
        default="bvbrc",
        help="Phenotype acquisition source for the acquire step"
    )
    parser.add_argument(
        "--ncbi-ast-file",
        type=str,
        default=None,
        help="Path to an NCBI AST Browser CSV/TSV export to import"
    )
    parser.add_argument(
        "--combine-sources",
        action="store_true",
        help="When acquiring from both sources, combine them into data/raw/ecoli_resistance.csv"
    )
    
    args = parser.parse_args()
    selected_results_source = (args.results_source or resolve_results_source(args.feature_source)).upper()
    results_layout = config.get_results_layout(selected_results_source)
    configure_pipeline_logging(results_layout["root"])
    
    logger.info("="*80)
    logger.info("E. COLI ANTIBIOTIC RESISTANCE PREDICTION PIPELINE")
    logger.info("="*80)
    logger.info(f"Steps to run: {args.steps}")
    logger.info(f"Antibiotic: {args.antibiotic}")
    logger.info(f"Feature source: {args.feature_source}")
    if args.feature_source == "genes":
        logger.info(f"Testing standard: {args.testing_standard}")
    if args.feature_source == "ncbi_genes" and args.split_group != "sample":
        logger.info(f"Strict split group: {args.split_group}")
    logger.info(f"Results source: {selected_results_source}")
    
    steps_to_run = args.steps
    if "all" in steps_to_run:
        steps_to_run = ["acquire", "preprocess", "explore", "train", "validate"]
    
    try:
        # Step 1: Data acquisition
        if "acquire" in steps_to_run:
            run_data_acquisition(
                skip_if_exists=args.skip_existing,
                acquisition_source=args.acquisition_source,
                ncbi_ast_file=args.ncbi_ast_file,
                combine_sources=args.combine_sources
            )
        
        # Step 2: Preprocessing
        if "preprocess" in steps_to_run:
            run_preprocessing()
        
        # Step 3: Exploratory analysis
        if "explore" in steps_to_run:
            run_exploratory_analysis(antibiotic=args.antibiotic)
        
        # Step 4: Model training
        trainer, X_train, X_test, y_train, y_test = None, None, None, None, None
        if "train" in steps_to_run:
            trainer, X_train, X_test, y_train, y_test = run_model_training_with_source(
                antibiotic=args.antibiotic,
                feature_source=args.feature_source,
                testing_standard=args.testing_standard,
                split_group=args.split_group,
                cv=config.CV_FOLDS,
                use_ensemble=True,
                results_source=selected_results_source
            )
        
        # Step 5: Model validation
        if "validate" in steps_to_run:
            if trainer is None:
                logger.warning("Training step was not run. Skipping validation.")
            else:
                run_model_validation(
                    trainer,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    results_source=selected_results_source
                )
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nResults saved to: {results_layout['root']}")
        logger.info(f"Models saved to: {config.MODEL_DIR}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
