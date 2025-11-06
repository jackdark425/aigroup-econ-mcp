"""
Microeconometrics Adapter for Econometrics MCP Tools
Provides unified interfaces for discrete choice, count data, and limited dependent variable models
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, List
import json
import logging

# Import microeconometrics modules
from econometrics.specific_data_modeling.micro_discrete_limited_data import (
    LogitModel,
    ProbitModel,
    MultinomialLogit,
    OrderedLogit,
    ConditionalLogit,
    PoissonModel,
    NegativeBinomialModel,
    ZeroInflatedPoissonModel,
    ZeroInflatedNegativeBinomialModel,
    TobitModel,
    HeckmanModel
)

from tools.data_loader import DataLoader

# Set up logging
logger = logging.getLogger(__name__)


def logit_adapter(
    X_data: Optional[Union[List[float], List[List[float]]]] = None,
    y_data: Optional[List[int]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    output_format: str = 'json',
    save_path: Optional[str] = None
) -> str:
    """Logistic regression adapter"""
    try:
        if file_path:
            data = DataLoader.load_from_file(file_path)
            X_data = data.get('x_data', data.get('X', data.get('features')))
            y_data = data.get('y_data', data.get('y', data.get('target')))
            if feature_names is None:
                feature_names = data.get('feature_names')
        
        if X_data is None or y_data is None:
            raise ValueError("X_data and y_data must be provided")
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        model = LogitModel()
        model.fit(X, y)
        
        results = model.results_
        formatted_results = {
            'model_type': 'logit',
            'coefficients': results.params.tolist(),
            'std_errors': results.bse.tolist(),
            'z_values': results.tvalues.tolist(),
            'p_values': results.pvalues.tolist(),
            'pseudo_r_squared': float(results.prsquared),
            'log_likelihood': float(results.llf),
            'aic': float(results.aic),
            'bic': float(results.bic),
            'n_obs': int(results.nobs),
            'feature_names': feature_names or [f'X{i+1}' for i in range(X.shape[1])]
        }
        
        return json.dumps(formatted_results, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Logit failed: {str(e)}")
        return json.dumps({'error': str(e)}, indent=2, ensure_ascii=False)


def probit_adapter(
    X_data: Optional[Union[List[float], List[List[float]]]] = None,
    y_data: Optional[List[int]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    output_format: str = 'json',
    save_path: Optional[str] = None
) -> str:
    """Probit regression adapter"""
    try:
        if file_path:
            data = DataLoader.load_from_file(file_path)
            X_data = data.get('x_data', data.get('X', data.get('features')))
            y_data = data.get('y_data', data.get('y', data.get('target')))
            if feature_names is None:
                feature_names = data.get('feature_names')
        
        if X_data is None or y_data is None:
            raise ValueError("X_data and y_data must be provided")
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        model = ProbitModel()
        model.fit(X, y)
        
        results = model.results_
        formatted_results = {
            'model_type': 'probit',
            'coefficients': results.params.tolist(),
            'std_errors': results.bse.tolist(),
            'z_values': results.tvalues.tolist(),
            'p_values': results.pvalues.tolist(),
            'pseudo_r_squared': float(results.prsquared),
            'log_likelihood': float(results.llf),
            'aic': float(results.aic),
            'bic': float(results.bic),
            'n_obs': int(results.nobs),
            'feature_names': feature_names or [f'X{i+1}' for i in range(X.shape[1])]
        }
        
        return json.dumps(formatted_results, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Probit failed: {str(e)}")
        return json.dumps({'error': str(e)}, indent=2, ensure_ascii=False)


def multinomial_logit_adapter(
    X_data: Optional[Union[List[float], List[List[float]]]] = None,
    y_data: Optional[List[int]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    output_format: str = 'json',
    save_path: Optional[str] = None
) -> str:
    """Multinomial Logit adapter"""
    try:
        if file_path:
            data = DataLoader.load_from_file(file_path)
            X_data = data.get('x_data', data.get('X', data.get('features')))
            y_data = data.get('y_data', data.get('y', data.get('target')))
            if feature_names is None:
                feature_names = data.get('feature_names')
        
        if X_data is None or y_data is None:
            raise ValueError("X_data and y_data must be provided")
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        model = MultinomialLogit()
        model.fit(X, y)
        
        results = model.results_
        formatted_results = {
            'model_type': 'multinomial_logit',
            'coefficients': results.params.tolist(),
            'std_errors': results.bse.tolist(),
            'z_values': results.tvalues.tolist(),
            'p_values': results.pvalues.tolist(),
            'pseudo_r_squared': float(results.prsquared),
            'log_likelihood': float(results.llf),
            'aic': float(results.aic),
            'bic': float(results.bic),
            'n_obs': int(results.nobs),
            'classes': model.classes_.tolist(),
            'feature_names': feature_names or [f'X{i+1}' for i in range(X.shape[1])]
        }
        
        return json.dumps(formatted_results, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Multinomial Logit failed: {str(e)}")
        return json.dumps({'error': str(e)}, indent=2, ensure_ascii=False)


def poisson_adapter(
    X_data: Optional[Union[List[float], List[List[float]]]] = None,
    y_data: Optional[List[int]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    output_format: str = 'json',
    save_path: Optional[str] = None
) -> str:
    """Poisson regression adapter"""
    try:
        if file_path:
            data = DataLoader.load_from_file(file_path)
            X_data = data.get('x_data', data.get('X', data.get('features')))
            y_data = data.get('y_data', data.get('y', data.get('target')))
            if feature_names is None:
                feature_names = data.get('feature_names')
        
        if X_data is None or y_data is None:
            raise ValueError("X_data and y_data must be provided")
        
        X = np.array(X_data)
        y = np.array(y_data).astype(int)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        model = PoissonModel()
        model.fit(X, y)
        
        results = model.results_
        formatted_results = {
            'model_type': 'poisson',
            'coefficients': results.params.tolist(),
            'std_errors': results.bse.tolist(),
            'z_values': results.tvalues.tolist(),
            'p_values': results.pvalues.tolist(),
            'pseudo_r_squared': float(results.prsquared),
            'log_likelihood': float(results.llf),
            'aic': float(results.aic),
            'bic': float(results.bic),
            'n_obs': int(results.nobs),
            'feature_names': feature_names or [f'X{i+1}' for i in range(X.shape[1])]
        }
        
        return json.dumps(formatted_results, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Poisson failed: {str(e)}")
        return json.dumps({'error': str(e)}, indent=2, ensure_ascii=False)


def negative_binomial_adapter(
    X_data: Optional[Union[List[float], List[List[float]]]] = None,
    y_data: Optional[List[int]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    distr: str = 'nb2',
    output_format: str = 'json',
    save_path: Optional[str] = None
) -> str:
    """Negative Binomial regression adapter"""
    try:
        if file_path:
            data = DataLoader.load_from_file(file_path)
            X_data = data.get('x_data', data.get('X', data.get('features')))
            y_data = data.get('y_data', data.get('y', data.get('target')))
            if feature_names is None:
                feature_names = data.get('feature_names')
        
        if X_data is None or y_data is None:
            raise ValueError("X_data and y_data must be provided")
        
        X = np.array(X_data)
        y = np.array(y_data).astype(int)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        model = NegativeBinomialModel(distr=distr)
        model.fit(X, y)
        
        results = model.results_
        formatted_results = {
            'model_type': 'negative_binomial',
            'distribution': distr,
            'coefficients': results.params.tolist(),
            'std_errors': results.bse.tolist(),
            'z_values': results.tvalues.tolist(),
            'p_values': results.pvalues.tolist(),
            'pseudo_r_squared': float(results.prsquared),
            'log_likelihood': float(results.llf),
            'aic': float(results.aic),
            'bic': float(results.bic),
            'n_obs': int(results.nobs),
            'feature_names': feature_names or [f'X{i+1}' for i in range(X.shape[1])]
        }
        
        return json.dumps(formatted_results, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Negative Binomial failed: {str(e)}")
        return json.dumps({'error': str(e)}, indent=2, ensure_ascii=False)


def tobit_adapter(
    X_data: Optional[Union[List[float], List[List[float]]]] = None,
    y_data: Optional[List[float]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    lower_bound: float = 0.0,
    upper_bound: Optional[float] = None,
    output_format: str = 'json',
    save_path: Optional[str] = None
) -> str:
    """Tobit model adapter"""
    try:
        if file_path:
            data = DataLoader.load_from_file(file_path)
            X_data = data.get('x_data', data.get('X', data.get('features')))
            y_data = data.get('y_data', data.get('y', data.get('target')))
            if feature_names is None:
                feature_names = data.get('feature_names')
        
        if X_data is None or y_data is None:
            raise ValueError("X_data and y_data must be provided")
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        model = TobitModel(lower_bound=lower_bound, upper_bound=upper_bound)
        model.fit(X, y)
        
        results = model.results_
        formatted_results = {
            'model_type': 'tobit',
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'coefficients': results.params.tolist(),
            'std_errors': results.bse.tolist(),
            'z_values': results.tvalues.tolist(),
            'p_values': results.pvalues.tolist(),
            'log_likelihood': float(results.llf),
            'aic': float(results.aic),
            'bic': float(results.bic),
            'n_obs': int(results.nobs),
            'feature_names': feature_names or [f'X{i+1}' for i in range(X.shape[1])]
        }
        
        return json.dumps(formatted_results, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Tobit failed: {str(e)}")
        return json.dumps({'error': str(e)}, indent=2, ensure_ascii=False)


def heckman_adapter(
    X_select_data: Optional[Union[List[float], List[List[float]]]] = None,
    Z_data: Optional[Union[List[float], List[List[float]]]] = None,
    y_data: Optional[List[float]] = None,
    s_data: Optional[List[int]] = None,
    file_path: Optional[str] = None,
    selection_feature_names: Optional[List[str]] = None,
    outcome_feature_names: Optional[List[str]] = None,
    output_format: str = 'json',
    save_path: Optional[str] = None
) -> str:
    """Heckman selection model adapter"""
    try:
        if file_path:
            data = DataLoader.load_from_file(file_path)
            X_select_data = data.get('X_select', data.get('selection_features'))
            Z_data = data.get('Z', data.get('outcome_features'))
            y_data = data.get('y', data.get('outcome'))
            s_data = data.get('s', data.get('selection'))
            if selection_feature_names is None:
                selection_feature_names = data.get('selection_feature_names')
            if outcome_feature_names is None:
                outcome_feature_names = data.get('outcome_feature_names')
        
        if X_select_data is None or Z_data is None or y_data is None or s_data is None:
            raise ValueError("All data must be provided")
        
        X_select = np.array(X_select_data)
        Z = np.array(Z_data)
        y = np.array(y_data)
        s = np.array(s_data).astype(int)
        
        if X_select.ndim == 1:
            X_select = X_select.reshape(-1, 1)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        
        model = HeckmanModel()
        model.fit(X_select, Z, y, s)
        
        selection_names = selection_feature_names or [f'SelectX{i+1}' for i in range(X_select.shape[1])]
        outcome_names = outcome_feature_names or [f'OutcomeZ{i+1}' for i in range(Z.shape[1])]
        
        formatted_results = {
            'model_type': 'heckman',
            'selection_results': {
                'coefficients': model.selection_results_.params.tolist(),
                'std_errors': model.selection_results_.bse.tolist(),
                'z_values': model.selection_results_.tvalues.tolist(),
                'p_values': model.selection_results_.pvalues.tolist(),
                'feature_names': selection_names
            },
            'outcome_results': {
                'coefficients': model.outcome_results_.params.tolist(),
                'std_errors': model.outcome_results_.bse.tolist(),
                't_values': model.outcome_results_.tvalues.tolist(),
                'p_values': model.outcome_results_.pvalues.tolist(),
                'feature_names': outcome_names
            },
            'n_obs': len(y),
            'n_selected': int(np.sum(s))
        }
        
        return json.dumps(formatted_results, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Heckman failed: {str(e)}")
        return json.dumps({'error': str(e)}, indent=2, ensure_ascii=False)