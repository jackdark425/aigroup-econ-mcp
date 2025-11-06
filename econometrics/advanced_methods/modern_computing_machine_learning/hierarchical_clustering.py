"""
Hierarchical Clustering implementation for econometric analysis
"""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, linkage_tree
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from typing import Union, Optional, Dict, Any

# 可选导入matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
except UnicodeDecodeError:
    # 处理编码问题
    MATPLOTLIB_AVAILABLE = False


class EconHierarchicalClustering:
    """
    Hierarchical Clustering for econometric analysis
    """
    
    def __init__(self, n_clusters: int = 2, linkage: str = 'ward', 
                 metric: str = 'euclidean'):
        """
        Initialize Hierarchical Clustering model
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to find
        linkage : str, 'ward', 'complete', 'average', 'single'
            Which linkage criterion to use
        metric : str or callable
            Metric used to compute the linkage. Can be 'euclidean', 'l1', 'l2',
            'manhattan', 'cosine', or 'precomputed'
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.scaler = StandardScaler()
        
        # Initialize model
        # Note: 'ward' linkage requires 'euclidean' metric
        if linkage == 'ward':
            self.metric = 'euclidean'
            
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric if linkage != 'ward' else 'euclidean'
        )
        
        # Store linkage matrix for dendrogram
        self.linkage_matrix = None
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'EconHierarchicalClustering':
        """
        Fit the Hierarchical Clustering model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : EconHierarchicalClustering
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        
        # Compute linkage matrix for dendrogram
        if self.metric != 'precomputed':
            distance_matrix = pdist(X_scaled, metric=self.metric)
            self.linkage_matrix = linkage(distance_matrix, method=self.linkage)
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame] = None) -> np.ndarray:
        """
        Get cluster labels
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features) or None
            Data to predict (not used in hierarchical clustering, 
            returns labels from fit)
            
        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        return self.model.labels_
    
    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit the hierarchical clustering model and return cluster labels
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        self.fit(X)
        return self.model.labels_
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, float]:
        """
        Evaluate clustering performance
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data to evaluate
            
        Returns:
        --------
        metrics : dict
            Dictionary with evaluation metrics
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        labels = self.model.labels_
        
        # Calculate metrics if more than 1 cluster
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X_scaled, labels)
            calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
        else:
            silhouette = 0.0
            calinski_harabasz = 0.0
        
        return {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz
        }
    
    def plot_dendrogram(self, X: Union[np.ndarray, pd.DataFrame] = None, 
                       truncate_mode: str = 'level', p: int = 5,
                       figsize: tuple = (12, 8)) -> Optional:
        """
        Plot dendrogram for hierarchical clustering
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features) or None
            Data to visualize (if None, uses data from fit)
        truncate_mode : str
            Truncation mode for dendrogram
        p : int
            Parameter for truncation
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : matplotlib Figure or None
            The figure object, or None if matplotlib is not available
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib is not available. Skipping visualization.")
            return None
            
        # Compute linkage matrix if not already computed
        if self.linkage_matrix is None and X is not None:
            X_scaled = self.scaler.transform(X)
            distance_matrix = pdist(X_scaled, metric=self.metric)
            self.linkage_matrix = linkage(distance_matrix, method=self.linkage)
        
        if self.linkage_matrix is None:
            raise ValueError("No linkage matrix available. Please fit the model first or provide data.")
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        dendrogram(
            self.linkage_matrix,
            truncate_mode=truncate_mode,
            p=p,
            ax=ax
        )
        ax.set_xlabel('Sample Index or (Cluster Size)')
        ax.set_ylabel('Distance')
        ax.set_title('Hierarchical Clustering Dendrogram')
        
        return fig


def hierarchical_clustering_analysis(X: Union[np.ndarray, pd.DataFrame],
                                   n_clusters: int = 2,
                                   linkage: str = 'ward',
                                   metric: str = 'euclidean') -> dict:
    """
    Perform complete Hierarchical Clustering analysis
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Features
    n_clusters : int
        Number of clusters to find
    linkage : str, 'ward', 'complete', 'average', 'single'
        Which linkage criterion to use
    metric : str or callable
        Metric used to compute the linkage
        
    Returns:
    --------
    results : dict
        Dictionary with model, cluster labels, and evaluation metrics
    """
    # Initialize and fit model
    hc_model = EconHierarchicalClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric
    )
    labels = hc_model.fit_predict(X)
    
    # Evaluate clustering
    metrics = hc_model.evaluate(X)
    
    return {
        'model': hc_model,
        'labels': labels,
        'metrics': metrics,
        'X': X
    }