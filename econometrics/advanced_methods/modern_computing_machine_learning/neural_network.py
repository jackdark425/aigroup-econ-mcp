"""
Neural Network implementation for econometric analysis
"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Union, Optional, List, Tuple


class EconNeuralNetwork:
    """
    Neural Network for econometric analysis with both regression and classification capabilities
    """
    
    def __init__(self, problem_type: str = 'regression', hidden_layer_sizes: tuple = (100,),
                 activation: str = 'relu', solver: str = 'adam', alpha: float = 0.0001,
                 learning_rate: str = 'constant', learning_rate_init: float = 0.001,
                 max_iter: int = 200, random_state: int = 42):
        """
        Initialize Neural Network model
        
        Parameters:
        -----------
        problem_type : str, 'regression' or 'classification'
            Type of problem to solve
        hidden_layer_sizes : tuple
            The ith element represents the number of neurons in the ith hidden layer
        activation : str, 'identity', 'logistic', 'tanh', 'relu'
            Activation function for the hidden layer
        solver : str, 'lbfgs', 'sgd', 'adam'
            The solver for weight optimization
        alpha : float
            L2 penalty (regularization term) parameter
        learning_rate : str, 'constant', 'invscaling', 'adaptive'
            Learning rate schedule for weight updates
        learning_rate_init : float
            The initial learning rate used
        max_iter : int
            Maximum number of iterations
        random_state : int
            Random state for reproducibility
        """
        self.problem_type = problem_type
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        if problem_type == 'regression':
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                alpha=alpha,
                learning_rate=learning_rate,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                random_state=random_state
            )
        elif problem_type == 'classification':
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                alpha=alpha,
                learning_rate=learning_rate,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                random_state=random_state
            )
        else:
            raise ValueError("problem_type must be either 'regression' or 'classification'")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'EconNeuralNetwork':
        """
        Fit the Neural Network model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : EconNeuralNetwork
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict using the Neural Network model
        
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
        Predict class probabilities using the Neural Network model (classification only)
        
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


def neural_network_analysis(X: Union[np.ndarray, pd.DataFrame], 
                           y: Union[np.ndarray, pd.Series],
                           problem_type: str = 'regression',
                           hidden_layer_sizes: tuple = (100,),
                           activation: str = 'relu',
                           solver: str = 'adam',
                           test_size: float = 0.2,
                           alpha: float = 0.0001,
                           learning_rate: str = 'constant',
                           learning_rate_init: float = 0.001,
                           max_iter: int = 200,
                           random_state: int = 42) -> dict:
    """
    Perform complete Neural Network analysis
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Features
    y : array-like of shape (n_samples,)
        Target variable
    problem_type : str, 'regression' or 'classification'
        Type of problem to solve
    hidden_layer_sizes : tuple
        The ith element represents the number of neurons in the ith hidden layer
    activation : str, 'identity', 'logistic', 'tanh', 'relu'
        Activation function for the hidden layer
    solver : str, 'lbfgs', 'sgd', 'adam'
        The solver for weight optimization
    test_size : float
        Proportion of dataset to include in test split
    alpha : float
        L2 penalty (regularization term) parameter
    learning_rate : str, 'constant', 'invscaling', 'adaptive'
        Learning rate schedule for weight updates
    learning_rate_init : float
        The initial learning rate used
    max_iter : int
        Maximum number of iterations
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
    nn_model = EconNeuralNetwork(
        problem_type=problem_type,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state
    )
    nn_model.fit(X_train, y_train)
    
    # Evaluate model
    train_results = nn_model.evaluate(X_train, y_train)
    test_results = nn_model.evaluate(X_test, y_test)
    
    # For classification, also get probabilities
    if problem_type == 'classification':
        train_proba = nn_model.predict_proba(X_train)
        test_proba = nn_model.predict_proba(X_test)
    else:
        train_proba = None
        test_proba = None
    
    return {
        'model': nn_model,
        'train_results': train_results,
        'test_results': test_results,
        'train_proba': train_proba,
        'test_proba': test_proba,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }