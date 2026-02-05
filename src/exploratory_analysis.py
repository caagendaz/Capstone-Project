"""
Exploratory Data Analysis Module for E. coli Antibiotic Resistance Project

Performs:
1. Principal Component Analysis (PCA) for dimensionality reduction
2. K-means clustering to identify patterns
3. Data visualization and exploration
4. Statistical summaries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Optional, List
import logging
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ExploratoryAnalyzer:
    """
    Performs exploratory data analysis on E. coli resistance data.
    """
    
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.pca = None
        self.kmeans = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load processed data."""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded data shape: {df.shape}")
        return df
    
    def explore_data_summary(self, df: pd.DataFrame):
        """
        Generate and log basic data summary statistics.
        """
        logger.info("\n" + "="*60)
        logger.info("DATA SUMMARY")
        logger.info("="*60)
        
        logger.info(f"\nDataset shape: {df.shape}")
        logger.info(f"Number of isolates: {df['isolate_id'].nunique() if 'isolate_id' in df.columns else 'N/A'}")
        
        if 'antibiotic' in df.columns:
            logger.info(f"\nAntibiotics tested: {df['antibiotic'].nunique()}")
            logger.info("\nSample counts per antibiotic:")
            logger.info(df['antibiotic'].value_counts().to_string())
        
        if 'resistance_phenotype' in df.columns:
            logger.info("\nResistance phenotype distribution:")
            logger.info(df['resistance_phenotype'].value_counts().to_string())
        
        if 'binary_resistant' in df.columns:
            resistant_pct = df['binary_resistant'].mean() * 100
            logger.info(f"\nOverall resistance rate: {resistant_pct:.2f}%")
        
        if 'testing_standard' in df.columns:
            logger.info("\nTesting standard distribution:")
            logger.info(df['testing_standard'].value_counts().to_string())
        
        logger.info("\nMissing values:")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            logger.info(missing.to_string())
        else:
            logger.info("No missing values found")
    
    def plot_resistance_distribution(self, df: pd.DataFrame, 
                                     antibiotic: Optional[str] = None):
        """
        Plot resistance phenotype distribution.
        """
        if antibiotic:
            df_plot = df[df['antibiotic'] == antibiotic]
            title = f"Resistance Distribution - {antibiotic}"
            filename = f"resistance_dist_{antibiotic.replace(' ', '_')}.png"
        else:
            df_plot = df
            title = "Overall Resistance Distribution"
            filename = "resistance_dist_overall.png"
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Categorical phenotypes
        if 'resistance_phenotype' in df_plot.columns:
            phenotype_counts = df_plot['resistance_phenotype'].value_counts()
            axes[0].bar(phenotype_counts.index, phenotype_counts.values, 
                       color=['green', 'orange', 'red'])
            axes[0].set_xlabel('Phenotype')
            axes[0].set_ylabel('Count')
            axes[0].set_title('Resistance Phenotypes (S/I/R)')
            axes[0].grid(axis='y', alpha=0.3)
        
        # Binary resistance
        if 'binary_resistant' in df_plot.columns:
            binary_counts = df_plot['binary_resistant'].value_counts()
            colors = ['green', 'red']
            axes[1].bar(['Susceptible', 'Resistant'], 
                       [binary_counts.get(0, 0), binary_counts.get(1, 0)],
                       color=colors)
            axes[1].set_ylabel('Count')
            axes[1].set_title('Binary Resistance (Resistant vs Susceptible)')
            axes[1].grid(axis='y', alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
        plt.close()
    
    def plot_antibiotic_comparison(self, df: pd.DataFrame):
        """
        Compare resistance rates across different antibiotics.
        """
        if 'antibiotic' not in df.columns or 'binary_resistant' not in df.columns:
            logger.warning("Cannot plot antibiotic comparison: missing columns")
            return
        
        # Calculate resistance rate per antibiotic
        resistance_rates = df.groupby('antibiotic')['binary_resistant'].agg([
            ('resistance_rate', 'mean'),
            ('count', 'count')
        ]).sort_values('resistance_rate', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(resistance_rates.index, resistance_rates['resistance_rate'] * 100)
        
        # Color bars by resistance rate
        colors = plt.cm.RdYlGn_r(resistance_rates['resistance_rate'])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Resistance Rate (%)')
        ax.set_ylabel('Antibiotic')
        ax.set_title('Resistance Rates by Antibiotic', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add sample counts
        for i, (idx, row) in enumerate(resistance_rates.iterrows()):
            ax.text(row['resistance_rate'] * 100 + 1, i, 
                   f"n={int(row['count'])}", 
                   va='center', fontsize=8)
        
        plt.tight_layout()
        output_path = self.output_dir / "antibiotic_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
        plt.close()
    
    def perform_pca(self, X: pd.DataFrame, 
                   n_components: int = 50,
                   variance_threshold: float = 0.95) -> Tuple[np.ndarray, PCA]:
        """
        Perform Principal Component Analysis for dimensionality reduction.
        
        Args:
            X: Feature matrix (AMR genes)
            n_components: Number of components to compute
            variance_threshold: Cumulative variance to retain
            
        Returns:
            Transformed data and fitted PCA object
        """
        logger.info(f"Performing PCA on data with shape {X.shape}...")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        n_components = min(n_components, min(X.shape))
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Calculate cumulative variance
        cumsum_var = np.cumsum(self.pca.explained_variance_ratio_)
        n_components_needed = np.argmax(cumsum_var >= variance_threshold) + 1
        
        logger.info(f"Explained variance by {n_components} components: "
                   f"{cumsum_var[-1]:.4f}")
        logger.info(f"Components needed for {variance_threshold*100}% variance: "
                   f"{n_components_needed}")
        
        return X_pca, self.pca
    
    def plot_pca_variance(self, pca: PCA):
        """
        Plot explained variance from PCA.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Variance per component
        axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1),
                   pca.explained_variance_ratio_)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Variance Explained by Each PC')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Cumulative variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        axes[1].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'b-', linewidth=2)
        axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        axes[1].axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Variance Explained')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "pca_variance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
        plt.close()
    
    def plot_pca_scatter(self, X_pca: np.ndarray, y: pd.Series,
                        title: str = "PCA Visualization"):
        """
        Plot first two principal components colored by resistance.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                           c=y, cmap='RdYlGn_r',
                           alpha=0.6, s=50)
        
        ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Resistance (0=Susceptible, 1=Resistant)')
        
        plt.tight_layout()
        output_path = self.output_dir / "pca_scatter.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
        plt.close()
    
    def find_optimal_clusters(self, X_pca: np.ndarray, 
                            max_clusters: int = 10) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette score.
        """
        logger.info("Finding optimal number of clusters...")
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_pca)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_pca, labels))
        
        # Plot metrics
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow plot
        axes[0].plot(k_range, inertias, 'bo-', linewidth=2)
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia (Within-cluster sum of squares)')
        axes[0].set_title('Elbow Method')
        axes[0].grid(alpha=0.3)
        
        # Silhouette score
        axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2)
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Score by k')
        axes[1].grid(alpha=0.3)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        axes[1].axvline(x=optimal_k, color='green', linestyle='--', 
                       label=f'Optimal k={optimal_k}')
        axes[1].legend()
        
        plt.tight_layout()
        output_path = self.output_dir / "optimal_clusters.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
        plt.close()
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def perform_clustering(self, X_pca: np.ndarray, 
                          n_clusters: int = 3) -> np.ndarray:
        """
        Perform K-means clustering.
        """
        logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_pca)
        
        # Calculate clustering metrics
        silhouette = silhouette_score(X_pca, cluster_labels)
        davies_bouldin = davies_bouldin_score(X_pca, cluster_labels)
        
        logger.info(f"Silhouette Score: {silhouette:.4f}")
        logger.info(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        
        # Cluster size distribution
        unique, counts = np.unique(cluster_labels, return_counts=True)
        logger.info("\nCluster sizes:")
        for cluster, count in zip(unique, counts):
            logger.info(f"  Cluster {cluster}: {count} samples")
        
        return cluster_labels
    
    def plot_clusters(self, X_pca: np.ndarray, cluster_labels: np.ndarray,
                     y: Optional[pd.Series] = None):
        """
        Visualize clusters in PCA space.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Colored by cluster
        scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                                  c=cluster_labels, cmap='viridis',
                                  alpha=0.6, s=50)
        axes[0].scatter(self.kmeans.cluster_centers_[:, 0],
                       self.kmeans.cluster_centers_[:, 1],
                       c='red', marker='X', s=200, edgecolors='black',
                       label='Centroids')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].set_title('K-means Clusters')
        axes[0].legend()
        plt.colorbar(scatter1, ax=axes[0], label='Cluster')
        
        # Plot 2: Colored by resistance if available
        if y is not None:
            scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                                      c=y, cmap='RdYlGn_r',
                                      alpha=0.6, s=50)
            axes[1].set_xlabel('PC1')
            axes[1].set_ylabel('PC2')
            axes[1].set_title('Resistance Phenotypes')
            plt.colorbar(scatter2, ax=axes[1], label='Resistant')
        
        plt.tight_layout()
        output_path = self.output_dir / "clusters_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
        plt.close()
    
    def analyze_cluster_resistance(self, cluster_labels: np.ndarray, 
                                   y: pd.Series):
        """
        Analyze resistance rates within each cluster.
        """
        logger.info("\nAnalyzing resistance rates by cluster...")
        
        df_analysis = pd.DataFrame({
            'cluster': cluster_labels,
            'resistant': y
        })
        
        cluster_stats = df_analysis.groupby('cluster')['resistant'].agg([
            ('resistance_rate', 'mean'),
            ('count', 'count')
        ])
        
        logger.info("\nResistance rates by cluster:")
        logger.info(cluster_stats.to_string())
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(cluster_stats.index, cluster_stats['resistance_rate'] * 100)
        
        # Color bars
        colors = plt.cm.RdYlGn_r(cluster_stats['resistance_rate'])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Resistance Rate (%)')
        ax.set_title('Resistance Rates by Cluster', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add counts
        for i, (idx, row) in enumerate(cluster_stats.iterrows()):
            ax.text(i, row['resistance_rate'] * 100 + 2,
                   f"n={int(row['count'])}\n{row['resistance_rate']*100:.1f}%",
                   ha='center', fontsize=10)
        
        plt.tight_layout()
        output_path = self.output_dir / "cluster_resistance_rates.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
        plt.close()


def main():
    """
    Main exploratory analysis pipeline.
    """
    analyzer = ExploratoryAnalyzer()
    
    # Example workflow
    logger.info("Starting exploratory analysis...")
    
    # Load processed data
    # df = analyzer.load_data("data/processed/merged_data_CLSI.csv")
    
    # Basic exploration
    # analyzer.explore_data_summary(df)
    # analyzer.plot_resistance_distribution(df)
    # analyzer.plot_antibiotic_comparison(df)
    
    # For specific antibiotic
    # antibiotic = "CIPROFLOXACIN"
    # df_abx = df[df['antibiotic'] == antibiotic]
    # 
    # # Prepare feature matrix
    # gene_cols = [col for col in df_abx.columns 
    #              if col not in ['isolate_id', 'antibiotic', 'mic_value',
    #                            'resistance_phenotype', 'testing_standard',
    #                            'binary_resistant', 'binary_resistant_liberal']]
    # X = df_abx[gene_cols]
    # y = df_abx['binary_resistant']
    # 
    # # PCA
    # X_pca, pca = analyzer.perform_pca(X, n_components=50)
    # analyzer.plot_pca_variance(pca)
    # analyzer.plot_pca_scatter(X_pca, y, f"PCA - {antibiotic}")
    # 
    # # Clustering
    # optimal_k = analyzer.find_optimal_clusters(X_pca)
    # cluster_labels = analyzer.perform_clustering(X_pca, n_clusters=optimal_k)
    # analyzer.plot_clusters(X_pca, cluster_labels, y)
    # analyzer.analyze_cluster_resistance(cluster_labels, y)
    
    logger.info("Exploratory analysis complete!")


if __name__ == "__main__":
    main()
