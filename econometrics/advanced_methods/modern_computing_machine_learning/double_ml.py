"""
Double Machine Learning implementation for causal inference
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from typing import Union, Optional, Dict, Any, Tuple
from scipy import stats


class DoubleML:
    """
    Double Machine Learning for causal inference with treatment effects
    """
    
    def __init__(self, learner_g: Any = None, learner_m: Any = None,
                 treatment_type: str = 'continuous', n_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize Double Machine Learning model
        
        Parameters:
        -----------
        learner_g : sklearn estimator, optional
            Estimator for the outcome regression (g)
            Default: RandomForestRegressor for continuous, RandomForestClassifier for binary
        learner_m : sklearn estimator, optional
            Estimator for the treatment regression (m)
            Default: RandomForestRegressor for continuous, RandomForestClassifier for binary
        treatment_type : str, 'continuous' or 'binary'
            Type of treatment variable
        n_folds : int
            Number of cross-fitting folds
        random_state : int
            Random state for reproducibility
        """
        self.learner_g = learner_g
        self.learner_m = learner_m
        self.treatment_type = treatment_type
        self.n_folds = n_folds
        self.random_state = random_state
        
        # Set default learners if not provided
        if self.learner_g is None:
            if treatment_type == 'continuous':
                self.learner_g = RandomForestRegressor(n_estimators=100, random_state=random_state)
            else:
                self.learner_g = RandomForestClassifier(n_estimators=100, random_state=random_state)
        
        if self.learner_m is None:
            if treatment_type == 'continuous':
                self.learner_m = RandomForestRegressor(n_estimators=100, random_state=random_state)
            else:
                self.learner_m = RandomForestClassifier(n_estimators=100, random_state=random_state)
        
        # Store results
        self.effect = None
        self.se = None
        self.ci = None
        self.pval = None
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            d: Union[np.ndarray, pd.Series]) -> 'DoubleML':
        """
        Fit the Double Machine Learning model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Covariates
        y : array-like of shape (n_samples,)
            Outcome variable
        d : array-like of shape (n_samples,)
            Treatment variable
            
        Returns:
        --------
        self : DoubleML
        """
        # Convert to numpy arrays if needed
        X = np.asarray(X)
        y = np.asarray(y)
        d = np.asarray(d)
        
        n_samples = X.shape[0]
        
        # Initialize arrays to store residuals
        y_res = np.zeros(n_samples)
        d_res = np.zeros(n_samples)
        
        # Create folds for cross-fitting
        np.random.seed(self.random_state)
        indices = np.random.permutation(n_samples)
        fold_size = n_samples // self.n_folds
        folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(self.n_folds)]
        # Add remaining samples to the last fold
        if n_samples % self.n_folds != 0:
            folds[-1] = np.concatenate([folds[-1], indices[self.n_folds*fold_size:]])
        
        # Cross-fitting
        for fold_idx, test_idx in enumerate(folds):
            # Training indices (all except test fold)
            train_idx = np.concatenate([folds[i] for i in range(self.n_folds) if i != fold_idx])
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            d_train, d_test = d[train_idx], d[test_idx]
            
            # Fit outcome regression and get residuals
            self.learner_g.fit(X_train, y_train)
            if self.treatment_type == 'continuous':
                y_pred = self.learner_g.predict(X_test)
            else:
                y_pred = self.learner_g.predict_proba(X_test)[:, 1]
            y_res[test_idx] = y_test - y_pred
            
            # Fit treatment regression and get residuals
            self.learner_m.fit(X_train, d_train)
            if self.treatment_type == 'continuous':
                d_pred = self.learner_m.predict(X_test)
            else:
                d_pred = self.learner_m.predict_proba(X_test)[:, 1]
            d_res[test_idx] = d_test - d_pred
        
        # Estimate treatment effect using partially linear regression
        # theta = E[d_res * y_res] / E[d_res^2]
        numerator = np.mean(d_res * y_res)
        denominator = np.mean(d_res**2)
        
        self.effect = numerator / denominator
        
        # Calculate standard error
        # Using the formula for the variance of the DML estimator
        residuals = y_res - self.effect * d_res
        variance = np.mean(residuals**2) / np.mean(d_res**2)**2 / n_samples
        self.se = np.sqrt(variance)
        
        # Calculate 95% confidence interval
        crit_val = 1.96  # 95% CI
        self.ci = (self.effect - crit_val * self.se, 
                   self.effect + crit_val * self.se)
        
        # Calculate p-value (two-sided test)
        z_score = self.effect / self.se
        # Use scipy.stats.norm for calculating p-value
        self.pval = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict treatment effects (constant for this implementation)
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples (not used, treatment effect is constant)
            
        Returns:
        --------
        effects : ndarray of shape (n_samples,)
            Estimated treatment effects
        """
        return np.full(X.shape[0], self.effect) if hasattr(X, 'shape') else np.full(len(X), self.effect)
    
    def get_effect(self) -> float:
        """
        Get the estimated treatment effect
        
        Returns:
        --------
        effect : float
            Estimated treatment effect
        """
        return self.effect
    
    def get_se(self) -> float:
        """
        Get the standard error of the treatment effect
        
        Returns:
        --------
        se : float
            Standard error of the treatment effect
        """
        return self.se
    
    def get_ci(self) -> Tuple[float, float]:
        """
        Get the 95% confidence interval for the treatment effect
        
        Returns:
        --------
        ci : tuple
            95% confidence interval (lower, upper)
        """
        return self.ci
    
    def get_pval(self) -> float:
        """
        Get the p-value for the treatment effect
        
        Returns:
        --------
        pval : float
            P-value for the treatment effect
        """
        return self.pval


def double_ml_analysis(X: Union[np.ndarray, pd.DataFrame], 
                      y: Union[np.ndarray, pd.Series],
                      d: Union[np.ndarray, pd.Series],
                      treatment_type: str = 'continuous',
                      n_folds: int = 5,
                      random_state: int = 42) -> dict:
    """
    Perform complete Double Machine Learning analysis
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Covariates
    y : array-like of shape (n_samples,)
        Outcome variable
    d : array-like of shape (n_samples,)
        Treatment variable
    treatment_type : str, 'continuous' or 'binary'
        Type of treatment variable
    n_folds : int
        Number of cross-fitting folds
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary with model and estimation results
    """
    # Initialize and fit model
    dml_model = DoubleML(
        treatment_type=treatment_type,
        n_folds=n_folds,
        random_state=random_state
    )
    dml_model.fit(X, y, d)
    
    # Get results
    effect = dml_model.get_effect()
    se = dml_model.get_se()
    ci = dml_model.get_ci()
    pval = dml_model.get_pval()
    
    return {
        'model': dml_model,
        'effect': effect,
        'se': se,
        'ci': ci,
        'pval': pval,
        'X': X,
        'y': y,
        'd': d
    }