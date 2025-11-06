"""
Gradient Boosting Machine (GBM/XGBoost) implementation for econometric analysis
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
from typing import Union, Optional, Dict, Any


class EconGradientBoosting:
    """
    Gradient Boosting for econometric analysis with both scikit-learn and XGBoost implementations
    """
    
    def __init__(self, algorithm: str = 'sklearn', problem_type: str = 'regression',
                 n_estimators: int = 100, learning_rate: float = 0.1, 
                 max_depth: int = 3, random_state: int = 42):
        """
        Initialize Gradient Boosting model
        
        Parameters:
        -----------
        algorithm : str, 'sklearn' or 'xgboost'
            Which implementation to use
        problem_type : str, 'regression' or 'classification'
            Type of problem to solve
        n_estimators : int
            Number of boosting stages
        learning_rate : float
            Learning rate shrinks the contribution of each tree
        max_depth : int
            Maximum depth of the individual regression estimators
        random_state : int
            Random state for reproducibility
        """
        self.algorithm = algorithm
        self.problem_type = problem_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        if algorithm == 'sklearn':
            if problem_type == 'regression':
                self.model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state
                )
            elif problem_type == 'classification':
                self.model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state
                )
        elif algorithm == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Please install it with 'pip install xgboost'")
            
            if problem_type == 'regression':
                self.model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state
                )
            elif problem_type == 'classification':
                self.model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state
                )
        else:
            raise ValueError("algorithm must be either 'sklearn' or 'xgboost'")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'EconGradientBoosting':
        """
        Fit the Gradient Boosting model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : EconGradientBoosting
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict using the Gradient Boosting model
        
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
    
    def feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importances
        
        Returns:
        --------
        importances : dict
            Dictionary with feature importances (depends on algorithm)
        """
        if self.algorithm == 'sklearn':
            return {
                'importances': self.model.feature_importances_
            }
        elif self.algorithm == 'xgboost':
            # XGBoost provides multiple importance types
            importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
            importances = {}
            for imp_type in importance_types:
                try:
                    importances[imp_type] = self.model.feature_importances_
                except:
                    pass
            return importances
    
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


def gradient_boosting_analysis(X: Union[np.ndarray, pd.DataFrame], 
                              y: Union[np.ndarray, pd.Series],
                              algorithm: str = 'sklearn',
                              problem_type: str = 'regression',
                              test_size: float = 0.2,
                              n_estimators: int = 100,
                              learning_rate: float = 0.1,
                              max_depth: int = 3,
                              random_state: int = 42) -> dict:
    """
    Perform complete Gradient Boosting analysis
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Features
    y : array-like of shape (n_samples,)
        Target variable
    algorithm : str, 'sklearn' or 'xgboost'
        Which implementation to use
    problem_type : str, 'regression' or 'classification'
        Type of problem to solve
    test_size : float
        Proportion of dataset to include in test split
    n_estimators : int
        Number of boosting stages
    learning_rate : float
        Learning rate shrinks the contribution of each tree
    max_depth : int
        Maximum depth of the individual regression estimators
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
    gb_model = EconGradientBoosting(
        algorithm=algorithm,
        problem_type=problem_type,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    gb_model.fit(X_train, y_train)
    
    # Evaluate model
    train_results = gb_model.evaluate(X_train, y_train)
    test_results = gb_model.evaluate(X_test, y_test)
    
    # Get feature importances
    importances = gb_model.feature_importance()
    
    return {
        'model': gb_model,
        'train_results': train_results,
        'test_results': test_results,
        'feature_importances': importances,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }