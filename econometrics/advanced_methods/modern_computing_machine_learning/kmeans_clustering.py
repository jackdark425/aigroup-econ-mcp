"""
K-Means Clustering implementation for econometric analysis
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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


class EconKMeans:
    """
    K-Means Clustering for econometric analysis
    """
    
    def __init__(self, n_clusters: int = 8, init: str = 'k-means++', n_init: int = 10,
                 max_iter: int = 300, random_state: int = 42, algorithm: str = 'lloyd',
                 use_minibatch: bool = False, batch_size: int = 1000):
        """
        Initialize K-Means clustering model
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to form
        init : str, 'k-means++', 'random'
            Method for initialization
        n_init : int
            Number of time the k-means algorithm will be run with different centroid seeds
        max_iter : int
            Maximum number of iterations of the k-means algorithm for a single run
        random_state : int
            Random state for reproducibility
        algorithm : str, 'lloyd', 'elkan'
            K-means algorithm to use
        use_minibatch : bool
            Whether to use MiniBatchKMeans for large datasets
        batch_size : int
            Size of the mini batches (only used when use_minibatch=True)
        """
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.algorithm = algorithm
        self.use_minibatch = use_minibatch
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        
        if use_minibatch:
            self.model = MiniBatchKMeans(
                n_clusters=n_clusters,
                init=init,
                max_iter=max_iter,
                random_state=random_state,
                batch_size=batch_size
            )
        else:
            self.model = KMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                random_state=random_state,
                algorithm=algorithm
            )
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'EconKMeans':
        """
        Fit the K-Means clustering model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : EconKMeans
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to predict
            
        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        # Scale features using the same scaler
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute cluster centers and predict cluster index for each sample
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        return self.model.fit_predict(X_scaled)
    
    def cluster_centers(self) -> np.ndarray:
        """
        Get the cluster centers
        
        Returns:
        --------
        centers : ndarray of shape (n_clusters, n_features)
            Coordinates of cluster centers
        """
        return self.scaler.inverse_transform(self.model.cluster_centers_)
    
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
        labels = self.model.predict(X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
        
        return {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'inertia': self.model.inertia_,
            'n_iter': self.model.n_iter_
        }
    
    def visualize_clusters(self, X: Union[np.ndarray, pd.DataFrame], 
                          max_features: int = 10, figsize: tuple = (12, 8)) -> Optional:
        """
        Visualize clusters using PCA for dimensionality reduction
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data to visualize
        max_features : int
            Maximum number of features to show in the plot
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
            
        # Scale features
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        
        # Use PCA for dimensionality reduction if there are more than 2 features
        if X_scaled.shape[1] > 2:
            pca = PCA(n_components=min(2, X_scaled.shape[1]))
            X_pca = pca.fit_transform(X_scaled)
        else:
            X_pca = X_scaled
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot points colored by cluster
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
        
        # Plot cluster centers if available in PCA space
        if hasattr(self.model, 'cluster_centers_'):
            centers_pca = pca.transform(self.model.cluster_centers_) if X_scaled.shape[1] > 2 else self.model.cluster_centers_
            ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='x', s=200, linewidths=3)
        
        ax.set_xlabel('Principal Component 1' if X_scaled.shape[1] > 2 else 'Feature 1')
        ax.set_ylabel('Principal Component 2' if X_scaled.shape[1] > 2 else 'Feature 2')
        ax.set_title('K-Means Clustering Results')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax)
        
        return fig


def kmeans_analysis(X: Union[np.ndarray, pd.DataFrame],
                   n_clusters: int = 8,
                   init: str = 'k-means++',
                   n_init: int = 10,
                   max_iter: int = 300,
                   random_state: int = 42,
                   algorithm: str = 'lloyd',
                   use_minibatch: bool = False,
                   batch_size: int = 1000) -> dict:
    """
    Perform complete K-Means clustering analysis
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Features
    n_clusters : int
        Number of clusters to form
    init : str, 'k-means++', 'random'
        Method for initialization
    n_init : int
        Number of time the k-means algorithm will be run with different centroid seeds
    max_iter : int
        Maximum number of iterations of the k-means algorithm for a single run
    random_state : int
        Random state for reproducibility
    algorithm : str, 'lloyd', 'elkan'
        K-means algorithm to use
    use_minibatch : bool
        Whether to use MiniBatchKMeans for large datasets
    batch_size : int
        Size of the mini batches (only used when use_minibatch=True)
        
    Returns:
    --------
    results : dict
        Dictionary with model, cluster labels, centers, and evaluation metrics
    """
    # Initialize and fit model
    kmeans_model = EconKMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        algorithm=algorithm,
        use_minibatch=use_minibatch,
        batch_size=batch_size
    )
    labels = kmeans_model.fit_predict(X)
    
    # Get cluster centers
    centers = kmeans_model.cluster_centers()
    
    # Evaluate clustering
    metrics = kmeans_model.evaluate(X)
    
    return {
        'model': kmeans_model,
        'labels': labels,
        'cluster_centers': centers,
        'metrics': metrics,
        'X': X
    }