"""
Data Acquisition Module for E. coli Antibiotic Resistance Project

This module handles downloading and initial processing of data from:
1. Kaggle dataset (E. coli resistance phenotypes)
2. Bacterial and Viral Bioinformatics Resource Center (genomic data via API)
"""

import argparse
import os
import requests
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KaggleDataDownloader:
    """
    Downloads E. coli resistance dataset from Kaggle.
    
    Note: Requires kaggle API credentials to be set up.
    Instructions: https://github.com/Kaggle/kaggle-api#api-credentials
    """
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self, dataset_name: str) -> Path:
        """
        Download dataset from Kaggle.
        
        Args:
            dataset_name: Kaggle dataset identifier (e.g., 'username/dataset-name')
            
        Returns:
            Path to downloaded file
        """
        try:
            import kaggle
            logger.info(f"Downloading dataset: {dataset_name}")
            kaggle.api.dataset_download_files(
                dataset_name,
                path=str(self.output_dir),
                unzip=True
            )
            logger.info(f"Dataset downloaded to {self.output_dir}")
            return self.output_dir
        except Exception as e:
            logger.error(f"Failed to download from Kaggle: {e}")
            logger.info("Please ensure Kaggle API is installed and configured.")
            logger.info("Run: pip install kaggle")
            logger.info("Setup credentials: https://github.com/Kaggle/kaggle-api")
            raise
            

class BVBRCDataDownloader:
    """
    Downloads E. coli data from BVBRC (Bacterial and Viral Bioinformatics Resource Center).
    NO API CREDENTIALS REQUIRED - completely free and open!
    
    BVBRC provides both:
    1. Resistance phenotypes (MIC values, S/I/R calls) - replaces Kaggle dataset
    2. AMR gene annotations - genomic data
    
    API Documentation: https://www.bv-brc.org/api/
    """
    
    def __init__(self, output_dir: str = "data/external"):
        self.base_url = "https://www.bv-brc.org/api"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        
    def search_genomes(self, 
                      organism: str = "Escherichia coli",
                      limit: int = 1000,
                      filters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Search for E. coli genomes in BVBRC database.
        
        Args:
            organism: Organism name to search for
            limit: Maximum number of results to return
            filters: Additional filters (e.g., {'isolation_country': 'USA'})
            
        Returns:
            DataFrame with genome metadata
        """
        endpoint = f"{self.base_url}/genome/"
        
        params = {
            'eq(organism_name,{})'.format(organism): '',
            'limit': limit,
            'http_accept': 'application/json'
        }
        
        if filters:
            for key, value in filters.items():
                params[f'eq({key},{value})'] = ''
        
        try:
            logger.info(f"Searching for {organism} genomes...")
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data)
            logger.info(f"Found {len(df)} genome records")
            return df
        except Exception as e:
            logger.error(f"Failed to search genomes: {e}")
            raise
    
    def get_amr_phenotypes(self, genome_ids: Optional[List[str]] = None, 
                          limit: int = 50000) -> pd.DataFrame:
        """
        Retrieve antibiotic resistance phenotype data (MIC, resistance calls).
        
        This gets the actual resistance testing results - equivalent to the Kaggle dataset!
        
        Args:
            genome_ids: Optional list of specific genome IDs to query
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with columns like:
            - genome_id, genome_name
            - antibiotic
            - resistant_phenotype (Susceptible/Resistant/Intermediate)
            - measurement_sign, measurement_value (MIC data)
            - laboratory_typing_method (testing method)
            - testing_standard (CLSI/EUCAST)
        """
        logger.info("Retrieving AMR phenotype data from BVBRC...")
        
        all_data = []
        
        # Get all E. coli AMR data using correct BVBRC POST request format
        try:
            # BVBRC API uses POST with RQL query in body
            endpoint = f"{self.base_url}/genome_amr/"
            
            # Use POST request with proper headers and RQL query
            headers = {
                'Content-Type': 'application/rqlquery+x-www-form-urlencoded',
                'Accept': 'application/json'
            }
            
            # RQL query for E. coli (taxon_id 562) AMR data
            rql_query = f"eq(taxon_id,562)&limit({limit})"
            
            logger.info(f"Fetching up to {limit} E. coli AMR records...")
            logger.info(f"Using RQL query: {rql_query}")
            
            response = self.session.post(endpoint, data=rql_query, headers=headers)
            response.raise_for_status()
            all_data = response.json()
            
        except Exception as e:
            logger.error(f"Failed to get AMR phenotypes via POST: {e}")
            # Fallback: try GET with select fields
            try:
                logger.info("Trying alternative GET request...")
                # Alternative: use solr-style parameters
                params = {
                    'keyword(Escherichia coli)': '',
                    'limit': limit,
                    'http_accept': 'application/json'
                }
                response = self.session.get(f"{self.base_url}/genome_amr/", params=params)
                response.raise_for_status()
                all_data = response.json()
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                raise
        
        if all_data:
            df = pd.DataFrame(all_data)
            logger.info(f"Retrieved {len(df)} AMR phenotype records")
            logger.info(f"Columns available: {df.columns.tolist()}")
            return df
        else:
            logger.warning("No AMR phenotype data retrieved")
            return pd.DataFrame()

    @staticmethod
    def _combine_unique_values(values: pd.Series) -> str:
        """Combine distinct non-empty values into a single semicolon-delimited string."""
        unique_values = []
        for value in values:
            if isinstance(value, list):
                value = "; ".join(str(item) for item in value if item)
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text and text not in unique_values:
                unique_values.append(text)
        return "; ".join(unique_values)

    def _fetch_amr_gene_source(self,
                               source_name: str,
                               sp_gene_endpoint: str,
                               headers: Dict[str, str],
                               genome_ids: List[str],
                               limit: int = 25000) -> List[Dict]:
        """Retrieve AMR gene annotations for one source and filter to target genomes."""
        records = []
        target_genomes = set(str(gid) for gid in genome_ids)
        rql_query = f"eq(taxon_id,562)&eq(source,{source_name})&limit({limit})"
        logger.info(f"Fetching {source_name} AMR data with query: {rql_query}")

        response = self.session.post(
            sp_gene_endpoint,
            data=rql_query,
            headers=headers,
            timeout=120
        )

        if response.status_code != 200:
            logger.warning("%s query returned status %s", source_name, response.status_code)
            return records

        data = response.json()
        logger.info("Retrieved %s %s AMR records", len(data), source_name)

        for record in data:
            genome_id = str(record.get('genome_id', ''))
            if genome_id not in target_genomes:
                continue

            records.append({
                'genome_id': genome_id,
                'gene': record.get('gene', record.get('source_id', '')),
                'product': record.get('product', record.get('function', '')),
                'patric_id': record.get('patric_id', record.get('feature_id', '')),
                'source': record.get('source', source_name),
                'evidence': record.get('evidence', ''),
                'classification': record.get('classification', [])
            })

        logger.info("Found %s %s annotations for target genomes", len(records), source_name)
        return records
    
    def get_amr_genes(self, genome_ids: List[str]) -> pd.DataFrame:
        """
        Retrieve antimicrobial resistance gene data for specific genomes.
        Uses the sp_gene endpoint which has curated AMR gene annotations
        from CARD (Comprehensive Antibiotic Resistance Database).
        
        Args:
            genome_ids: List of genome IDs to query
            
        Returns:
            DataFrame with AMR gene presence/absence data
        """
        logger.info(f"Retrieving AMR gene data for {len(genome_ids)} genomes...")
        
        all_amr_data = []
        
        # Query multiple curated sources and union them before building features
        logger.info("Querying sp_gene endpoint for curated AMR annotations...")
        sp_gene_endpoint = f"{self.base_url}/sp_gene/"

        headers = {
            'Content-Type': 'application/rqlquery+x-www-form-urlencoded',
            'Accept': 'application/json'
        }

        for source_name in ["CARD", "NDARO"]:
            try:
                all_amr_data.extend(
                    self._fetch_amr_gene_source(
                        source_name,
                        sp_gene_endpoint=sp_gene_endpoint,
                        headers=headers,
                        genome_ids=genome_ids
                    )
                )
            except Exception as e:
                logger.warning("%s query failed: %s", source_name, e)
        
        if all_amr_data:
            df = pd.DataFrame(all_amr_data)
            df['classification'] = df['classification'].apply(
                lambda value: "; ".join(str(item) for item in value if item)
                if isinstance(value, list) else value
            )
            df = (
                df.groupby(['genome_id', 'gene'], as_index=False)
                  .agg({
                      'product': self._combine_unique_values,
                      'patric_id': self._combine_unique_values,
                      'source': self._combine_unique_values,
                      'evidence': self._combine_unique_values,
                      'classification': self._combine_unique_values
                  })
            )
            logger.info(f"Final: {len(df)} unique AMR gene annotations from {df['genome_id'].nunique()} genomes")
            logger.info(f"Unique genes found: {df['gene'].nunique()}")
            if 'source' in df.columns:
                logger.info(f"Data sources: {df['source'].value_counts().to_dict()}")
            return df
        else:
            logger.warning("No AMR gene data retrieved from any endpoint - will use phenotype data for modeling")
            return pd.DataFrame(columns=['genome_id', 'gene', 'product', 'patric_id', 'source', 'evidence', 'classification'])
    
    def download_fasta(self, genome_id: str, output_file: Optional[str] = None) -> Path:
        """
        Download FASTA file for a specific genome.
        
        Args:
            genome_id: BVBRC genome ID
            output_file: Optional output filename
            
        Returns:
            Path to downloaded FASTA file
        """
        endpoint = f"{self.base_url}/genome_sequence/"
        
        if output_file is None:
            output_file = f"{genome_id}.fasta"
        
        output_path = self.output_dir / output_file
        
        try:
            params = {
                'eq(genome_id,{})'.format(genome_id): '',
                'http_accept': 'application/fasta'
            }
            
            logger.info(f"Downloading FASTA for genome {genome_id}")
            response = self.session.get(endpoint, params=params, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"FASTA saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to download FASTA for {genome_id}: {e}")
            raise
    
    def save_amr_data(self, amr_df: pd.DataFrame, filename: str = "amr_genes.csv"):
        """Save AMR gene data to CSV."""
        output_path = self.output_dir / filename
        amr_df.to_csv(output_path, index=False)
        logger.info(f"AMR data saved to {output_path}")
        return output_path


class NCBIASTDataImporter:
    """
    Imports AST Browser exports from NCBI Pathogen Detection / BioSample.

    NCBI documents CSV/TSV export from the AST Browser, but does not expose a
    stable public download API in the same way BVBRC does. This importer
    normalizes those exports into the project's phenotype schema.
    """

    COLUMN_MAP = {
        '#BioSample': 'biosample_accession',
        'Biosample': 'biosample_accession',
        'BioSample': 'biosample_accession',
        'BioProject': 'bioproject_accession',
        'Scientific name': 'scientific_name',
        'Scientific Name': 'scientific_name',
        'Antibiotic': 'antibiotic',
        'Resistance phenotype': 'resistant_phenotype',
        'Measurement sign': 'measurement_sign',
        'MIC (mg/L)': 'mic_value',
        'Disk diffusion (mm)': 'disk_diffusion_mm',
        'Laboratory typing platform': 'laboratory_typing_platform',
        'Vendor': 'vendor',
        'Laboratory typing method version or reagent': 'laboratory_typing_method_version',
        'Testing standard': 'testing_standard',
        'Collection date': 'collection_date',
        'Create date': 'date_inserted',
        'Location': 'location',
        'Host': 'host',
        'Isolation source': 'isolation_source',
        'Isolation type': 'isolation_type',
        'Isolate': 'ncbi_isolate_accession',
        'Organism group': 'organism_group'
    }

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read_export(filepath: str) -> pd.DataFrame:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"NCBI AST export not found: {filepath}")

        suffix = path.suffix.lower()
        sep = '\t' if suffix in {'.tsv', '.txt'} else ','
        df = pd.read_csv(path, sep=sep, low_memory=False)
        logger.info("Loaded %s NCBI AST rows from %s", len(df), filepath)
        return df

    @staticmethod
    def _standardize_phenotype(values: pd.Series) -> pd.Series:
        return (
            values.astype("string")
            .str.strip()
            .str.upper()
            .replace({
                'RESISTANT': 'R',
                'SUSCEPTIBLE': 'S',
                'INTERMEDIATE': 'I',
                'NON-SUSCEPTIBLE': 'R',
                'NONSUSCEPTIBLE': 'R'
            })
        )

    @staticmethod
    def _infer_lab_method(df: pd.DataFrame) -> pd.Series:
        if 'laboratory_typing_method' in df.columns:
            existing = df['laboratory_typing_method'].astype("string").str.strip()
            if existing.notna().any():
                return existing

        has_mic = pd.to_numeric(df.get('mic_value'), errors='coerce').notna()
        has_disk = pd.to_numeric(df.get('disk_diffusion_mm'), errors='coerce').notna()

        return pd.Series(
            np.where(
                has_mic,
                'BROTH DILUTION',
                np.where(has_disk, 'DISK DIFFUSION', pd.NA)
            ),
            index=df.index,
            dtype="string"
        )

    def normalize_ast_export(self, df: pd.DataFrame) -> pd.DataFrame:
        renamed = df.rename(columns=self.COLUMN_MAP).copy()

        biosample = renamed.get('biosample_accession', pd.Series(pd.NA, index=renamed.index, dtype="string"))
        isolate_acc = renamed.get('ncbi_isolate_accession', pd.Series(pd.NA, index=renamed.index, dtype="string"))
        scientific_name = renamed.get('scientific_name', pd.Series('Escherichia coli', index=renamed.index, dtype="string"))

        mic_values = pd.to_numeric(renamed.get('mic_value'), errors='coerce')
        disk_values = pd.to_numeric(renamed.get('disk_diffusion_mm'), errors='coerce')
        has_mic = mic_values.notna()

        normalized = pd.DataFrame({
            'genome_id': isolate_acc.fillna(biosample),
            'genome_name': scientific_name,
            'isolate_id': biosample.fillna(isolate_acc),
            'biosample_accession': biosample,
            'ncbi_isolate_accession': isolate_acc,
            'bioproject_accession': renamed.get('bioproject_accession'),
            'antibiotic': renamed.get('antibiotic'),
            'resistant_phenotype': self._standardize_phenotype(
                renamed.get('resistant_phenotype', pd.Series(pd.NA, index=renamed.index, dtype="string"))
            ),
            'measurement_sign': renamed.get('measurement_sign'),
            'measurement_value': np.where(has_mic, mic_values, disk_values),
            'mic_value': mic_values,
            'disk_diffusion_mm': disk_values,
            'measurement_unit': np.where(has_mic, 'mg/L', np.where(disk_values.notna(), 'mm', pd.NA)),
            'laboratory_typing_method': self._infer_lab_method(renamed),
            'laboratory_typing_platform': renamed.get('laboratory_typing_platform'),
            'laboratory_typing_method_version': renamed.get('laboratory_typing_method_version'),
            'testing_standard': renamed.get('testing_standard'),
            'date_inserted': renamed.get('date_inserted'),
            'collection_date': renamed.get('collection_date'),
            'location': renamed.get('location'),
            'host': renamed.get('host'),
            'isolation_source': renamed.get('isolation_source'),
            'isolation_type': renamed.get('isolation_type'),
            'organism_group': renamed.get('organism_group'),
            'source_database': 'NCBI_AST_BROWSER'
        })

        normalized = normalized[
            normalized['genome_name'].astype("string").str.contains('Escherichia coli', case=False, na=False)
        ].copy()

        if 'antibiotic' in normalized.columns:
            normalized['antibiotic'] = normalized['antibiotic'].astype("string").str.strip().str.upper()

        normalized = normalized.dropna(subset=['isolate_id', 'antibiotic', 'resistant_phenotype'])
        logger.info(
            "Normalized NCBI AST export to %s rows across %s isolates and %s antibiotics",
            len(normalized),
            normalized['isolate_id'].nunique(),
            normalized['antibiotic'].nunique()
        )
        return normalized

    def import_ast_export(self, filepath: str) -> pd.DataFrame:
        raw_df = self._read_export(filepath)
        return self.normalize_ast_export(raw_df)

    def save_phenotype_data(self, df: pd.DataFrame, filename: str = "ecoli_resistance_ncbi.csv") -> Path:
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info("NCBI phenotype data saved to %s", output_path)
        return output_path


def main():
    """
    Main function to download all required data from BVBRC.
    No API keys needed!
    """
    logger.info("Starting data acquisition from BVBRC...")
    logger.info("No API credentials required - BVBRC is fully open access!")
    
    bvbrc_downloader = BVBRCDataDownloader()
    
    # Step 1: Get resistance phenotype data (replaces Kaggle dataset)
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Downloading Resistance Phenotypes")
    logger.info("="*60)
    
    phenotypes_df = bvbrc_downloader.get_amr_phenotypes(limit=50000)
    
    if not phenotypes_df.empty:
        # Save to raw data directory
        output_path = Path("data/raw/ecoli_resistance.csv")
        phenotypes_df.to_csv(output_path, index=False)
        logger.info(f"Resistance phenotypes saved to {output_path}")
        logger.info(f"Total records: {len(phenotypes_df)}")
        
        if 'antibiotic' in phenotypes_df.columns:
            logger.info(f"Antibiotics tested: {phenotypes_df['antibiotic'].nunique()}")
        if 'genome_id' in phenotypes_df.columns:
            logger.info(f"Unique isolates: {phenotypes_df['genome_id'].nunique()}")
    else:
        logger.error("Failed to download phenotype data")
        return
    
    # Step 2: Get genome metadata
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Downloading Genome Metadata")
    logger.info("="*60)
    
    # Get unique genome IDs from phenotype data
    if 'genome_id' in phenotypes_df.columns:
        unique_genome_ids = phenotypes_df['genome_id'].unique()[:1000]  # Limit for efficiency
        logger.info(f"Found {len(unique_genome_ids)} unique genomes with resistance data")
    else:
        # Fallback: search for E. coli genomes
        genomes_df = bvbrc_downloader.search_genomes("Escherichia coli", limit=1000)
        genomes_df.to_csv("data/external/genome_metadata.csv", index=False)
        logger.info("Genome metadata saved")
        unique_genome_ids = genomes_df['genome_id'].tolist() if 'genome_id' in genomes_df.columns else []
    
    # Step 3: Get AMR gene data
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Downloading AMR Gene Annotations")
    logger.info("="*60)
    
    if len(unique_genome_ids) > 0:
        amr_genes_df = bvbrc_downloader.get_amr_genes(unique_genome_ids[:500])  # Limit for initial run
        
        if not amr_genes_df.empty:
            bvbrc_downloader.save_amr_data(amr_genes_df, "amr_genes.csv")
        else:
            logger.warning("No AMR gene data retrieved")
    
    logger.info("\n" + "="*60)
    logger.info("DATA ACQUISITION COMPLETE!")
    logger.info("="*60)
    logger.info("\nDownloaded:")
    logger.info(f"  ✓ Resistance phenotypes: data/raw/ecoli_resistance.csv")
    logger.info(f"  ✓ AMR genes: data/external/amr_genes.csv")
    logger.info(f"  ✓ Genome metadata: data/external/genome_metadata.csv")
    logger.info("\nNo API keys were needed - thank you BVBRC! 🎉")
 

def cli_main():
    """
    Run source-aware phenotype acquisition.
    """
    parser = argparse.ArgumentParser(description="Phenotype/genotype data acquisition")
    parser.add_argument("--source", choices=["bvbrc", "ncbi", "both"], default="bvbrc")
    parser.add_argument("--ncbi-ast-file", type=str, default=None)
    parser.add_argument("--combine-sources", action="store_true")
    args = parser.parse_args()

    logger.info("Starting data acquisition...")
    phenotype_frames = []

    if args.source in {"bvbrc", "both"}:
        logger.info("Downloading BVBRC phenotypes and genes...")
        bvbrc_downloader = BVBRCDataDownloader()
        phenotypes_df = bvbrc_downloader.get_amr_phenotypes(limit=50000)

        if phenotypes_df.empty:
            logger.warning("No BVBRC phenotype data retrieved")
        else:
            phenotypes_df = phenotypes_df.copy()
            phenotypes_df["source_database"] = "BVBRC"
            phenotypes_df.to_csv("data/raw/ecoli_resistance_bvbrc.csv", index=False)
            phenotype_frames.append(phenotypes_df)
            logger.info("Saved BVBRC phenotypes to data/raw/ecoli_resistance_bvbrc.csv")

            if 'genome_id' in phenotypes_df.columns:
                unique_genome_ids = phenotypes_df['genome_id'].unique()
                amr_genes_df = bvbrc_downloader.get_amr_genes(unique_genome_ids)
                if not amr_genes_df.empty:
                    bvbrc_downloader.save_amr_data(amr_genes_df, "amr_genes.csv")

    if args.source in {"ncbi", "both"}:
        if not args.ncbi_ast_file:
            raise ValueError("--ncbi-ast-file is required when --source includes ncbi")

        logger.info("Importing NCBI AST export from %s", args.ncbi_ast_file)
        importer = NCBIASTDataImporter(output_dir="data/raw")
        ncbi_df = importer.import_ast_export(args.ncbi_ast_file)
        importer.save_phenotype_data(ncbi_df, "ecoli_resistance_ncbi.csv")
        phenotype_frames.append(ncbi_df)

    if len(phenotype_frames) == 1:
        phenotype_frames[0].to_csv("data/raw/ecoli_resistance.csv", index=False)
        logger.info("Saved canonical phenotype file to data/raw/ecoli_resistance.csv")
    elif len(phenotype_frames) > 1 and args.combine_sources:
        combined_df = pd.concat(phenotype_frames, ignore_index=True, sort=False).drop_duplicates()
        combined_df.to_csv("data/raw/ecoli_resistance.csv", index=False)
        logger.warning("Saved combined phenotype file to data/raw/ecoli_resistance.csv after exact dedup only")
    elif len(phenotype_frames) > 1:
        logger.info("Source-specific phenotype files saved; canonical combined file not updated")


if __name__ == "__main__":
    cli_main()
