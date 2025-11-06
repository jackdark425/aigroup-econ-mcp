"""
Random Forest implementation for econometric analysis
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from typing import Union, Optional, Tuple


class EconRandomForest:
    """
    Random Forest for econometric analysis with both regression and classification capabilities
    """
    
    def __init__(self, problem_type: str = 'regression', n_estimators: int = 100, 
                 max_depth: Optional[int] = None, random_state: int = 42):
        """
        Initialize Random Forest model
        
        Parameters:
        -----------
        problem_type : str, 'regression' or 'classification'
            Type of problem to solve
        n_estimators : int
            Number of trees in the forest
        max_depth : int, optional
            Maximum depth of the tree
        random_state : int
            Random state for reproducibility
        """
        self.problem_type = problem_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        if problem_type == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        elif problem_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            raise ValueError("problem_type must be either 'regression' or 'classification'")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'EconRandomForest':
        """
        Fit the Random Forest model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : EconRandomForest
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict using the Random Forest model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        return self.model.predict(X)
    
    def feature_importance(self) -> np.ndarray:
        """
        Get feature importances
        
        Returns:
        --------
        importances : ndarray of shape (n_features,)
            Feature importances
        """
        return self.model.feature_importances_
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                 y: Union[np.ndarray, pd.Series]) -> dict:
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test data
        y : array-like of shape (n_samples,)
            True values
            
        Returns:
        --------
        metrics : dict
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        
        if self.problem_type == 'regression':
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            return {
                'mse': mse,
                'rmse': rmse,
                'predictions': y_pred
            }
        else:
            accuracy = accuracy_score(y, y_pred)
            return {
                'accuracy': accuracy,
                'predictions': y_pred
            }


def random_forest_analysis(X: Union[np.ndarray, pd.DataFrame], 
                          y: Union[np.ndarray, pd.Series],
                          problem_type: str = 'regression',
                          test_size: float = 0.2,
                          n_estimators: int = 100,
                          max_depth: Optional[int] = None,
                          random_state: int = 42) -> dict:
    """
    Perform complete Random Forest analysis
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Features
    y : array-like of shape (n_samples,)
        Target variable
    problem_type : str, 'regression' or 'classification'
        Type of problem to solve
    test_size : float
        Proportion of dataset to include in test split
    n_estimators : int
        Number of trees in the forest
    max_depth : int, optional
        Maximum depth of the tree
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary with model, predictions, and feature importances
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize and fit model
    rf_model = EconRandomForest(
        problem_type=problem_type,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    train_results = rf_model.evaluate(X_train, y_train)
    test_results = rf_model.evaluate(X_test, y_test)
    
    # Get feature importances
    importances = rf_model.feature_importance()
    
    return {
        'model': rf_model,
        'train_results': train_results,
        'test_results': test_results,
        'feature_importances': importances,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }