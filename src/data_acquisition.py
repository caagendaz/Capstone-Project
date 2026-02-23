"""
Data Acquisition Module for E. coli Antibiotic Resistance Project

This module handles downloading and initial processing of data from:
1. Kaggle dataset (E. coli resistance phenotypes)
2. Bacterial and Viral Bioinformatics Resource Center (genomic data via API)
"""

import os
import requests
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging

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
                          limit: int = 15000) -> pd.DataFrame:
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
        
        # Method 1: Query sp_gene endpoint for CARD AMR annotations
        logger.info("Querying sp_gene endpoint for CARD AMR annotations...")
        sp_gene_endpoint = f"{self.base_url}/sp_gene/"
        
        # Get all CARD AMR annotations for E. coli (taxon_id 562)
        # Then filter to only the genomes we're interested in
        try:
            headers = {
                'Content-Type': 'application/rqlquery+x-www-form-urlencoded',
                'Accept': 'application/json'
            }
            
            # Query CARD source for E. coli AMR genes
            rql_query = "eq(taxon_id,562)&eq(source,CARD)&limit(25000)"
            logger.info(f"Fetching CARD AMR data with query: {rql_query}")
            
            response = self.session.post(sp_gene_endpoint, data=rql_query, headers=headers, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Retrieved {len(data)} CARD AMR records")
                
                # Convert genome_ids to set for fast lookup
                target_genomes = set(str(gid) for gid in genome_ids)
                
                for record in data:
                    genome_id = str(record.get('genome_id', ''))
                    if genome_id in target_genomes:
                        all_amr_data.append({
                            'genome_id': genome_id,
                            'gene': record.get('gene', record.get('source_id', '')),
                            'product': record.get('product', record.get('function', '')),
                            'patric_id': record.get('patric_id', record.get('feature_id', '')),
                            'source': record.get('source', 'CARD'),
                            'evidence': record.get('evidence', ''),
                            'classification': record.get('classification', [])
                        })
                
                logger.info(f"Found {len(all_amr_data)} AMR annotations for target genomes")
            else:
                logger.warning(f"sp_gene query returned status {response.status_code}")
                
        except Exception as e:
            logger.warning(f"sp_gene query failed: {e}")
        
        # Method 2: Also try NDARO source as backup
        if len(all_amr_data) < 100:
            logger.info("Trying NDARO source as additional data source...")
            try:
                rql_query = "eq(taxon_id,562)&eq(source,NDARO)&limit(25000)"
                response = self.session.post(sp_gene_endpoint, data=rql_query, headers=headers, timeout=120)
                
                if response.status_code == 200:
                    data = response.json()
                    target_genomes = set(str(gid) for gid in genome_ids)
                    
                    for record in data:
                        genome_id = str(record.get('genome_id', ''))
                        if genome_id in target_genomes:
                            all_amr_data.append({
                                'genome_id': genome_id,
                                'gene': record.get('gene', record.get('source_id', '')),
                                'product': record.get('product', record.get('function', '')),
                                'patric_id': record.get('patric_id', ''),
                                'source': record.get('source', 'NDARO'),
                                'evidence': record.get('evidence', ''),
                                'classification': record.get('classification', [])
                            })
                    logger.info(f"Added NDARO data, total: {len(all_amr_data)} records")
            except Exception as e:
                logger.debug(f"NDARO query failed: {e}")
        
        if all_amr_data:
            df = pd.DataFrame(all_amr_data)
            # Remove duplicates
            df = df.drop_duplicates(subset=['genome_id', 'gene'])
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
    
    phenotypes_df = bvbrc_downloader.get_amr_phenotypes(limit=15000)
    
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


if __name__ == "__main__":
    main()
