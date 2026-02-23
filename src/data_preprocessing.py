"""
Data Preprocessing Module for E. coli Antibiotic Resistance Project

Handles:
1. Data cleaning (missing values, outliers)
2. Handling different testing standards (CLSI vs EUCAST)
3. Merging phenotype and genotype data
4. Building feature matrix for AMR genes (presence/absence)
5. Data splitting and stratification
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EcoliDataPreprocessor:
    """
    Preprocesses E. coli resistance and genomic data for modeling.
    """
    
    def __init__(self, raw_data_dir: str = "data/raw", 
                 processed_data_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_resistance_data(self, filepath: str) -> pd.DataFrame:
        """
        Load resistance phenotype data from CSV.
        
        Expected columns:
        - isolate_id: unique identifier
        - antibiotic: antibiotic name
        - mic_value: Minimum Inhibitory Concentration
        - resistance_phenotype: S (susceptible), I (intermediate), R (resistant)
        - testing_standard: CLSI or EUCAST
        """
        logger.info(f"Loading resistance data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} resistance records")
        logger.info(f"Columns: {df.columns.tolist()}")
        return df
    
    def load_genomic_data(self, filepath: str) -> pd.DataFrame:
        """
        Load genomic/AMR gene data.
        
        Expected columns:
        - isolate_id or genome_id: unique identifier
        - gene_name: AMR gene name
        - presence: 1 (present) or 0 (absent)
        """
        logger.info(f"Loading genomic data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} genomic records")
        return df
    
    def clean_resistance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean resistance phenotype data.
        
        Handles BVBRC data format with columns:
        - genome_id, genome_name, antibiotic, resistant_phenotype, 
        - measurement_value, testing_standard, etc.
        
        Steps:
        1. Remove duplicates
        2. Handle missing values
        3. Standardize categorical values
        4. Filter valid resistance phenotypes
        """
        logger.info("Cleaning resistance data...")
        initial_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_count - len(df)} duplicate rows")
        
        # Standardize resistance phenotypes (BVBRC uses 'resistant_phenotype')
        if 'resistant_phenotype' in df.columns:
            df['resistant_phenotype'] = df['resistant_phenotype'].str.strip().str.upper()
            
            # Map BVBRC phenotypes to standard S/I/R format
            phenotype_map = {
                'SUSCEPTIBLE': 'S',
                'INTERMEDIATE': 'I',
                'RESISTANT': 'R',
                'SUSCEPTIBLE-DOSE DEPENDENT': 'I',  # Treat SDD as intermediate
                'NON-SUSCEPTIBLE': 'R'  # Treat non-susceptible as resistant
            }
            df['resistance_phenotype'] = df['resistant_phenotype'].map(phenotype_map)
            
            # Remove rows where mapping failed
            df = df.dropna(subset=['resistance_phenotype'])
            logger.info(f"After phenotype mapping: {len(df)} records")
        elif 'resistance_phenotype' in df.columns:
            # Already has the correct column name
            df['resistance_phenotype'] = df['resistance_phenotype'].str.upper()
            phenotype_map = {
                'SUSCEPTIBLE': 'S',
                'INTERMEDIATE': 'I',
                'RESISTANT': 'R'
            }
            df['resistance_phenotype'] = df['resistance_phenotype'].replace(phenotype_map)
        
        # Handle MIC values (BVBRC uses 'measurement_value')
        mic_col = 'measurement_value' if 'measurement_value' in df.columns else 'mic_value'
        if mic_col in df.columns:
            missing_mic = df[mic_col].isna().sum()
            logger.info(f"Missing MIC values: {missing_mic} ({missing_mic/len(df)*100:.2f}%)")
            # Create a standardized mic_value column if needed
            if mic_col != 'mic_value':
                df['mic_value'] = df[mic_col]
        
        # Standardize antibiotic names
        if 'antibiotic' in df.columns:
            df['antibiotic'] = df['antibiotic'].str.strip().str.upper()
        
        # Standardize testing standards
        if 'testing_standard' in df.columns:
            df['testing_standard'] = df['testing_standard'].str.upper()
            # Keep only valid standards
            valid_standards = df['testing_standard'].isin(['CLSI', 'EUCAST'])
            if valid_standards.sum() > 0:
                df = df[valid_standards]
            logger.info(f"Testing standards: {df['testing_standard'].value_counts().to_dict()}")
        
        # Create isolate_id from genome_id if needed
        if 'genome_id' in df.columns and 'isolate_id' not in df.columns:
            df['isolate_id'] = df['genome_id']
        
        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df
    
    def split_by_testing_standard(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split dataset by testing standard (CLSI vs EUCAST).
        
        This addresses the different interpretation criteria between standards.
        """
        logger.info("Splitting data by testing standard...")
        
        splits = {}
        if 'testing_standard' in df.columns:
            for standard in df['testing_standard'].unique():
                splits[standard] = df[df['testing_standard'] == standard].copy()
                logger.info(f"{standard}: {len(splits[standard])} records")
        else:
            logger.warning("No testing_standard column found")
            splits['all'] = df
        
        return splits
    
    def filter_common_antibiotics(self, df: pd.DataFrame, 
                                  min_samples: int = 100) -> pd.DataFrame:
        """
        Filter to keep only commonly tested antibiotics.
        
        Args:
            df: Resistance data
            min_samples: Minimum number of samples required per antibiotic
        """
        logger.info("Filtering for common antibiotics...")
        
        if 'antibiotic' not in df.columns:
            logger.warning("No antibiotic column found")
            return df
        
        antibiotic_counts = df['antibiotic'].value_counts()
        common_antibiotics = antibiotic_counts[antibiotic_counts >= min_samples].index
        
        logger.info(f"Found {len(common_antibiotics)} antibiotics with >={min_samples} samples")
        logger.info(f"Common antibiotics: {common_antibiotics.tolist()}")
        
        df_filtered = df[df['antibiotic'].isin(common_antibiotics)]
        logger.info(f"Filtered to {len(df_filtered)} records")
        
        return df_filtered
    
    def build_amr_feature_matrix(self, genomic_df: pd.DataFrame, 
                                 isolate_ids: List[str]) -> pd.DataFrame:
        """
        Build binary feature matrix for AMR gene presence/absence.
        
        Args:
            genomic_df: DataFrame with AMR gene data (columns: genome_id/isolate_id, gene/gene_name)
            isolate_ids: List of isolate/genome IDs to include
            
        Returns:
            DataFrame with isolate_id as index and genes as columns (binary)
        """
        logger.info("Building AMR gene feature matrix...")
        
        # Handle different column naming conventions
        id_col = 'genome_id' if 'genome_id' in genomic_df.columns else 'isolate_id'
        gene_col = 'gene' if 'gene' in genomic_df.columns else 'gene_name'
        
        # Convert IDs to strings for matching
        genomic_df = genomic_df.copy()
        genomic_df[id_col] = genomic_df[id_col].astype(str)
        isolate_ids = [str(i) for i in isolate_ids]
        
        # Filter to relevant isolates
        genomic_df = genomic_df[genomic_df[id_col].isin(isolate_ids)]
        
        if genomic_df.empty:
            logger.warning("No matching isolates found in genomic data")
            return pd.DataFrame()
        
        # Add presence column if not exists (presence = 1 for all records)
        if 'presence' not in genomic_df.columns:
            genomic_df['presence'] = 1
        
        # Pivot to create feature matrix
        feature_matrix = genomic_df.pivot_table(
            index=id_col,
            columns=gene_col,
            values='presence',
            fill_value=0,
            aggfunc='max'  # In case of duplicates, take max
        )
        
        # Rename index to standard name
        feature_matrix.index.name = 'isolate_id'
        
        logger.info(f"Feature matrix shape: {feature_matrix.shape}")
        logger.info(f"Number of isolates: {len(feature_matrix)}")
        logger.info(f"Number of AMR genes: {len(feature_matrix.columns)}")
        
        return feature_matrix
    
    def merge_phenotype_genotype(self, 
                                resistance_df: pd.DataFrame,
                                feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Merge resistance phenotype data with genomic feature matrix.
        
        Args:
            resistance_df: Cleaned resistance data
            feature_matrix: AMR gene feature matrix (index is isolate/genome ID)
            
        Returns:
            Merged DataFrame ready for modeling
        """
        logger.info("Merging phenotype and genotype data...")
        
        # Determine which ID column to use for merging
        if 'isolate_id' in resistance_df.columns:
            merge_col = 'isolate_id'
        elif 'genome_id' in resistance_df.columns:
            merge_col = 'genome_id'
        else:
            logger.error("No ID column found in resistance data")
            return resistance_df
        
        # Ensure feature matrix index is string type for matching
        feature_matrix = feature_matrix.copy()
        feature_matrix.index = feature_matrix.index.astype(str)
        
        # Convert merge column to string
        resistance_df = resistance_df.copy()
        resistance_df[merge_col] = resistance_df[merge_col].astype(str)
        
        # Merge on the ID column
        merged = resistance_df.merge(
            feature_matrix,
            left_on=merge_col,
            right_index=True,
            how='inner'
        )
        
        logger.info(f"Merged dataset shape: {merged.shape}")
        logger.info(f"Number of records: {len(merged)}")
        
        if len(merged) == 0:
            logger.warning("No matches found during merge - check ID formats")
            logger.info(f"Resistance IDs sample: {resistance_df[merge_col].head().tolist()}")
            logger.info(f"Feature matrix IDs sample: {list(feature_matrix.index[:5])}")
        
        return merged
    
    def create_binary_labels(self, df: pd.DataFrame, 
                           phenotype_col: str = 'resistance_phenotype') -> pd.DataFrame:
        """
        Convert resistance phenotypes to binary classification.
        
        Options:
        1. R vs S/I (Resistant vs Not Resistant)
        2. R/I vs S (Resistant/Intermediate vs Susceptible)
        """
        logger.info("Creating binary resistance labels...")
        
        # Option 1: R vs S/I (more conservative - treat intermediate as susceptible)
        df['binary_resistant'] = (df[phenotype_col] == 'R').astype(int)
        
        # Option 2: R/I vs S (more liberal - treat intermediate as resistant)
        df['binary_resistant_liberal'] = df[phenotype_col].isin(['R', 'I']).astype(int)
        
        # Log class distribution
        logger.info(f"Binary resistance (R vs S/I):")
        logger.info(f"\n{df['binary_resistant'].value_counts()}")
        logger.info(f"\nClass balance: {df['binary_resistant'].mean():.2%} resistant")
        
        return df
    
    def prepare_modeling_data(self, 
                            df: pd.DataFrame,
                            antibiotic: str,
                            test_size: float = 0.2,
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                             pd.Series, pd.Series]:
        """
        Prepare data for modeling for a specific antibiotic.
        
        Args:
            df: Merged phenotype-genotype data
            antibiotic: Antibiotic to model
            test_size: Fraction of data for test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Preparing modeling data for {antibiotic}...")
        
        # Filter to specific antibiotic
        df_antibiotic = df[df['antibiotic'] == antibiotic].copy()
        logger.info(f"Records for {antibiotic}: {len(df_antibiotic)}")
        
        # Separate features and labels
        # Assume gene columns start after metadata columns
        metadata_cols = ['isolate_id', 'antibiotic', 'mic_value', 
                        'resistance_phenotype', 'testing_standard',
                        'binary_resistant', 'binary_resistant_liberal']
        
        gene_cols = [col for col in df_antibiotic.columns 
                    if col not in metadata_cols]
        
        X = df_antibiotic[gene_cols]
        y = df_antibiotic['binary_resistant']
        
        logger.info(f"Feature dimensions: {X.shape}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to CSV."""
        output_path = self.processed_data_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        return output_path
    
    def build_resistance_profile_features(self, df: pd.DataFrame, 
                                          target_antibiotic: str) -> pd.DataFrame:
        """
        Build feature matrix from resistance profiles to other antibiotics.
        
        For each isolate, create features based on resistance to other antibiotics.
        This is useful when genomic data is not available.
        
        Args:
            df: Cleaned resistance data with genome_id, antibiotic, resistance_phenotype
            target_antibiotic: The antibiotic we want to predict resistance for
            
        Returns:
            DataFrame with genome_id as index and resistance to other antibiotics as features
        """
        logger.info(f"Building resistance profile features for {target_antibiotic}...")
        
        # Ensure we have required columns
        id_col = 'genome_id' if 'genome_id' in df.columns else 'isolate_id'
        
        if id_col not in df.columns or 'antibiotic' not in df.columns:
            logger.error("Required columns not found")
            return pd.DataFrame()
        
        # Get other antibiotics (not the target)
        other_antibiotics = df[df['antibiotic'] != target_antibiotic]['antibiotic'].unique()
        logger.info(f"Using {len(other_antibiotics)} other antibiotics as features")
        
        # Create pivot table: rows = isolates, columns = antibiotics, values = resistance (R=1, else=0)
        df_copy = df.copy()
        df_copy['is_resistant'] = (df_copy['resistance_phenotype'] == 'R').astype(int)
        
        # Pivot to create resistance profile
        resistance_profile = df_copy.pivot_table(
            index=id_col,
            columns='antibiotic',
            values='is_resistant',
            fill_value=np.nan,  # Use NaN for missing tests
            aggfunc='max'  # Take max if multiple tests
        )
        
        # Remove the target antibiotic column (will be our label)
        if target_antibiotic in resistance_profile.columns:
            # Save target as label
            target_labels = resistance_profile[target_antibiotic]
            resistance_profile = resistance_profile.drop(columns=[target_antibiotic])
        else:
            logger.warning(f"Target antibiotic {target_antibiotic} not in data")
            target_labels = None
        
        # Handle missing values - impute with column median
        resistance_profile = resistance_profile.fillna(resistance_profile.median())
        
        logger.info(f"Feature matrix shape: {resistance_profile.shape}")
        logger.info(f"Features: {resistance_profile.columns.tolist()[:10]}...")
        
        return resistance_profile, target_labels
    
    def prepare_modeling_data_from_profiles(self,
                                           df: pd.DataFrame,
                                           target_antibiotic: str,
                                           test_size: float = 0.2,
                                           random_state: int = 42) -> Tuple:
        """
        Prepare data for modeling using resistance profiles as features.
        
        Args:
            df: Cleaned resistance data
            target_antibiotic: Antibiotic to predict
            test_size: Fraction of data for test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Preparing modeling data from resistance profiles for {target_antibiotic}...")
        
        # Build features from resistance profiles
        X, y = self.build_resistance_profile_features(df, target_antibiotic)
        
        if X.empty or y is None or y.empty:
            logger.error("Could not build feature matrix")
            return None, None, None, None
        
        # Align X and y (keep only isolates that have both features and labels)
        common_ids = X.index.intersection(y.dropna().index)
        X = X.loc[common_ids]
        y = y.loc[common_ids]
        
        # Remove isolates with all NaN features
        valid_rows = ~X.isna().all(axis=1)
        X = X[valid_rows]
        y = y[valid_rows]
        
        # Fill any remaining NaN with 0
        X = X.fillna(0)
        
        logger.info(f"Final feature dimensions: {X.shape}")
        logger.info(f"Class distribution: Resistant={y.sum()}, Susceptible={len(y)-y.sum()}")
        
        if len(X) < 50:
            logger.error("Not enough samples for training")
            return None, None, None, None
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
        return output_path


def main():
    """
    Main preprocessing pipeline.
    """
    preprocessor = EcoliDataPreprocessor()
    
    # Example workflow - adjust file paths as needed
    logger.info("Starting preprocessing pipeline...")
    
    # 1. Load data
    # resistance_df = preprocessor.load_resistance_data("data/raw/ecoli_resistance.csv")
    # genomic_df = preprocessor.load_genomic_data("data/external/amr_genes.csv")
    
    # 2. Clean data
    # resistance_df = preprocessor.clean_resistance_data(resistance_df)
    
    # 3. Filter common antibiotics
    # resistance_df = preprocessor.filter_common_antibiotics(resistance_df, min_samples=100)
    
    # 4. Split by testing standard
    # data_splits = preprocessor.split_by_testing_standard(resistance_df)
    
    # 5. For each split, build feature matrix and merge
    # for standard, df in data_splits.items():
    #     isolate_ids = df['isolate_id'].unique()
    #     feature_matrix = preprocessor.build_amr_feature_matrix(genomic_df, isolate_ids)
    #     merged_df = preprocessor.merge_phenotype_genotype(df, feature_matrix)
    #     merged_df = preprocessor.create_binary_labels(merged_df)
    #     preprocessor.save_processed_data(merged_df, f"merged_data_{standard}.csv")
    
    logger.info("Preprocessing pipeline complete!")


if __name__ == "__main__":
    main()
