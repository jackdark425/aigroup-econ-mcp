"""
Support Vector Machine (SVM) implementation for econometric analysis
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Union, Optional, Dict, Any


class EconSVM:
    """
    Support Vector Machine for econometric analysis with both regression and classification capabilities
    """
    
    def __init__(self, problem_type: str = 'regression', kernel: str = 'rbf',
                 C: float = 1.0, gamma: str = 'scale', random_state: int = 42):
        """
        Initialize SVM model
        
        Parameters:
        -----------
        problem_type : str, 'regression' or 'classification'
            Type of problem to solve
        kernel : str, 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
            Specifies the kernel type to be used in the algorithm
        C : float
            Regularization parameter
        gamma : str or float, 'scale' or 'auto' or float
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        random_state : int
            Random state for reproducibility (used in probability estimation)
        """
        self.problem_type = problem_type
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        if problem_type == 'regression':
            self.model = SVR(
                kernel=kernel,
                C=C,
                gamma=gamma
            )
        elif problem_type == 'classification':
            self.model = SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                random_state=random_state,
                probability=True
            )
        else:
            raise ValueError("problem_type must be either 'regression' or 'classification'")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'EconSVM':
        """
        Fit the SVM model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : EconSVM
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict using the SVM model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        # Scale features using the same scaler
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities using the SVM model (classification only)
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        y_proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if self.problem_type != 'classification':
            raise ValueError("predict_proba is only available for classification problems")
        
        # Scale features using the same scaler
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
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


def svm_analysis(X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series],
                problem_type: str = 'regression',
                kernel: str = 'rbf',
                test_size: float = 0.2,
                C: float = 1.0,
                gamma: str = 'scale',
                random_state: int = 42) -> dict:
    """
    Perform complete SVM analysis
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Features
    y : array-like of shape (n_samples,)
        Target variable
    problem_type : str, 'regression' or 'classification'
        Type of problem to solve
    kernel : str, 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
        Specifies the kernel type to be used in the algorithm
    test_size : float
        Proportion of dataset to include in test split
    C : float
        Regularization parameter
    gamma : str or float, 'scale' or 'auto' or float
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary with model, predictions, and evaluation metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize and fit model
    svm_model = EconSVM(
        problem_type=problem_type,
        kernel=kernel,
        C=C,
        gamma=gamma,
        random_state=random_state
    )
    svm_model.fit(X_train, y_train)
    
    # Evaluate model
    train_results = svm_model.evaluate(X_train, y_train)
    test_results = svm_model.evaluate(X_test, y_test)
    
    # For classification, also get probabilities
    if problem_type == 'classification':
        train_proba = svm_model.predict_proba(X_train)
        test_proba = svm_model.predict_proba(X_test)
    else:
        train_proba = None
        test_proba = None
    
    return {
        'model': svm_model,
        'train_results': train_results,
        'test_results': test_results,
        'train_proba': train_proba,
        'test_proba': test_proba,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }