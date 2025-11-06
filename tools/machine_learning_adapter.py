"""
Machine Learning Adapter for Econometrics MCP Tools
Provides unified interfaces for 8 machine learning models
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, List
import json
import logging

# Import econometrics machine learning modules
from econometrics.advanced_methods.modern_computing_machine_learning import (
    random_forest_analysis,
    gradient_boosting_analysis,
    svm_analysis,
    neural_network_analysis,
    kmeans_analysis,
    hierarchical_clustering_analysis,
    double_ml_analysis,
    causal_forest_analysis
)

from tools.data_loader import DataLoader
from tools.output_formatter import OutputFormatter

# Set up logging
logger = logging.getLogger(__name__)


def convert_to_serializable(obj: Any) -> Any:
    """
    递归转换numpy数组和其他不可序列化对象为JSON可序列化格式
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj


def format_output(results: Dict[str, Any], output_format: str = 'json', save_path: Optional[str] = None) -> str:
    """
    统一的输出格式化函数
    参考OLS适配器的实现方式
    
    Args:
        results: 结果字典
        output_format: 输出格式 ('json', 'markdown', 'text')
        save_path: 保存路径（可选）
    
    Returns:
        格式化后的字符串结果
    """
    # 转换所有numpy数组为可序列化格式
    serializable_results = convert_to_serializable(results)
    
    if output_format == 'json':
        json_result = json.dumps(serializable_results, ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        # 对于非JSON格式，直接返回JSON（机器学习结果暂不支持Markdown格式化）
        json_result = json.dumps(serializable_results, ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result


def random_forest_adapter(
    X_data: Optional[Union[List[float], List[List[float]]]] = None,
    y_data: Optional[List[float]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    problem_type: str = 'regression',
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    output_format: str = 'json',
    save_path: Optional[str] = None
) -> str:
    """Random Forest analysis adapter"""
    try:
        if file_path:
            data = DataLoader.load_from_file(file_path)
            if isinstance(data, dict):
                X_data = data.get('X', data.get('features'))
                y_data = data.get('y', data.get('target'))
                if feature_names is None:
                    feature_names = data.get('feature_names')
        
        if X_data is None or y_data is None:
            raise ValueError("X_data and y_data must be provided or loaded from file")
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        results = random_forest_analysis(
            X=X, y=y, problem_type=problem_type, test_size=test_size,
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        
        formatted_results = {
            'model_type': 'random_forest',
            'problem_type': problem_type,
            'train_results': results['train_results'],
            'test_results': results['test_results'],
            'feature_importances': results['feature_importances'].tolist(),
            'feature_names': feature_names,
            'model_parameters': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'test_size': test_size,
                'random_state': random_state
            }
        }
        
        return format_output(formatted_results, output_format, save_path)
        
    except Exception as e:
        logger.error(f"Random Forest failed: {str(e)}")
        return format_output({'error': str(e)}, output_format)


def gradient_boosting_adapter(
    X_data: Optional[Union[List[float], List[List[float]]]] = None,
    y_data: Optional[List[float]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    algorithm: str = 'sklearn',
    problem_type: str = 'regression',
    test_size: float = 0.2,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int = 42,
    output_format: str = 'json',
    save_path: Optional[str] = None
) -> str:
    """Gradient Boosting analysis adapter"""
    try:
        if file_path:
            data = DataLoader.load_from_file(file_path)
            if isinstance(data, dict):
                X_data = data.get('X', data.get('features'))
                y_data = data.get('y', data.get('target'))
                if feature_names is None:
                    feature_names = data.get('feature_names')
        
        if X_data is None or y_data is None:
            raise ValueError("X_data and y_data must be provided")
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        results = gradient_boosting_analysis(
            X=X, y=y, algorithm=algorithm, problem_type=problem_type,
            test_size=test_size, n_estimators=n_estimators,
            learning_rate=learning_rate, max_depth=max_depth, random_state=random_state
        )
        
        formatted_results = {
            'model_type': 'gradient_boosting',
            'algorithm': algorithm,
            'problem_type': problem_type,
            'train_results': results['train_results'],
            'test_results': results['test_results'],
            'feature_importances': results['feature_importances'].tolist() if hasattr(results['feature_importances'], 'tolist') else results['feature_importances'],
            'feature_names': feature_names,
            'model_parameters': {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'test_size': test_size
            }
        }
        
        return format_output(formatted_results, output_format, save_path)
        
    except Exception as e:
        logger.error(f"Gradient Boosting failed: {str(e)}")
        return format_output({'error': str(e)}, output_format)


def svm_adapter(
    X_data: Optional[Union[List[float], List[List[float]]]] = None,
    y_data: Optional[List[float]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    problem_type: str = 'regression',
    kernel: str = 'rbf',
    test_size: float = 0.2,
    C: float = 1.0,
    gamma: str = 'scale',
    random_state: int = 42,
    output_format: str = 'json',
    save_path: Optional[str] = None
) -> str:
    """SVM analysis adapter"""
    try:
        if file_path:
            data = DataLoader.load_from_file(file_path)
            if isinstance(data, dict):
                X_data = data.get('X', data.get('features'))
                y_data = data.get('y', data.get('target'))
                if feature_names is None:
                    feature_names = data.get('feature_names')
        
        if X_data is None or y_data is None:
            raise ValueError("X_data and y_data must be provided")
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        results = svm_analysis(
            X=X, y=y, problem_type=problem_type, kernel=kernel,
            test_size=test_size, C=C, gamma=gamma, random_state=random_state
        )
        
        formatted_results = {
            'model_type': 'svm',
            'problem_type': problem_type,
            'kernel': kernel,
            'train_results': results['train_results'],
            'test_results': results['test_results'],
            'feature_names': feature_names,
            'model_parameters': {
                'C': C,
                'gamma': gamma,
                'test_size': test_size
            }
        }
        
        if problem_type == 'classification':
            formatted_results['train_proba_shape'] = results['train_proba'].shape if results['train_proba'] is not None else None
            formatted_results['test_proba_shape'] = results['test_proba'].shape if results['test_proba'] is not None else None
        
        return format_output(formatted_results, output_format, save_path)
        
    except Exception as e:
        logger.error(f"SVM failed: {str(e)}")
        return format_output({'error': str(e)}, output_format)


def neural_network_adapter(
    X_data: Optional[Union[List[float], List[List[float]]]] = None,
    y_data: Optional[List[float]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    problem_type: str = 'regression',
    hidden_layer_sizes: tuple = (100,),
    activation: str = 'relu',
    solver: str = 'adam',
    test_size: float = 0.2,
    alpha: float = 0.0001,
    learning_rate: str = 'constant',
    learning_rate_init: float = 0.001,
    max_iter: int = 200,
    random_state: int = 42,
    output_format: str = 'json',
    save_path: Optional[str] = None
) -> str:
    """Neural Network analysis adapter"""
    try:
        if file_path:
            data = DataLoader.load_from_file(file_path)
            if isinstance(data, dict):
                X_data = data.get('X', data.get('features'))
                y_data = data.get('y', data.get('target'))
                if feature_names is None:
                    feature_names = data.get('feature_names')
        
        if X_data is None or y_data is None:
            raise ValueError("X_data and y_data must be provided")
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        results = neural_network_analysis(
            X=X, y=y, problem_type=problem_type, hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, solver=solver, test_size=test_size, alpha=alpha,
            learning_rate=learning_rate, learning_rate_init=learning_rate_init,
            max_iter=max_iter, random_state=random_state
        )
        
        formatted_results = {
            'model_type': 'neural_network',
            'problem_type': problem_type,
            'train_results': results['train_results'],
            'test_results': results['test_results'],
            'feature_names': feature_names,
            'model_parameters': {
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': activation,
                'solver': solver,
                'alpha': alpha,
                'learning_rate': learning_rate,
                'learning_rate_init': learning_rate_init,
                'max_iter': max_iter
            }
        }
        
        if problem_type == 'classification':
            formatted_results['train_proba_shape'] = results['train_proba'].shape if results['train_proba'] is not None else None
            formatted_results['test_proba_shape'] = results['test_proba'].shape if results['test_proba'] is not None else None
        
        return format_output(formatted_results, output_format, save_path)
        
    except Exception as e:
        logger.error(f"Neural Network failed: {str(e)}")
        return format_output({'error': str(e)}, output_format)


def kmeans_clustering_adapter(
    X_data: Optional[Union[List[float], List[List[float]]]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    n_clusters: int = 8,
    init: str = 'k-means++',
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
    algorithm: str = 'lloyd',
    use_minibatch: bool = False,
    batch_size: int = 1000,
    output_format: str = 'json',
    save_path: Optional[str] = None
) -> str:
    """K-Means Clustering analysis adapter"""
    try:
        if file_path:
            data = DataLoader.load_from_file(file_path)
            if isinstance(data, dict):
                X_data = data.get('X', data.get('features'))
                if feature_names is None:
                    feature_names = data.get('feature_names')
        
        if X_data is None:
            raise ValueError("X_data must be provided")
        
        X = np.array(X_data)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        results = kmeans_analysis(
            X=X, n_clusters=n_clusters, init=init, n_init=n_init,
            max_iter=max_iter, random_state=random_state, algorithm=algorithm,
            use_minibatch=use_minibatch, batch_size=batch_size
        )
        
        formatted_results = {
            'model_type': 'kmeans_clustering',
            'labels': results['labels'].tolist(),
            'cluster_centers': results['cluster_centers'].tolist(),
            'metrics': results['metrics'],
            'feature_names': feature_names,
            'model_parameters': {
                'n_clusters': n_clusters,
                'init': init,
                'n_init': n_init,
                'max_iter': max_iter,
                'algorithm': algorithm,
                'use_minibatch': use_minibatch
            }
        }
        
        return format_output(formatted_results, output_format, save_path)
        
    except Exception as e:
        logger.error(f"K-Means Clustering failed: {str(e)}")
        return format_output({'error': str(e)}, output_format)


def hierarchical_clustering_adapter(
    X_data: Optional[Union[List[float], List[List[float]]]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    n_clusters: int = 2,
    linkage: str = 'ward',
    metric: str = 'euclidean',
    output_format: str = 'json',
    save_path: Optional[str] = None
) -> str:
    """Hierarchical Clustering analysis adapter"""
    try:
        if file_path:
            data = DataLoader.load_from_file(file_path)
            if isinstance(data, dict):
                X_data = data.get('X', data.get('features'))
                if feature_names is None:
                    feature_names = data.get('feature_names')
        
        if X_data is None:
            raise ValueError("X_data must be provided")
        
        X = np.array(X_data)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        results = hierarchical_clustering_analysis(
            X=X, n_clusters=n_clusters, linkage=linkage, metric=metric
        )
        
        formatted_results = {
            'model_type': 'hierarchical_clustering',
            'labels': results['labels'].tolist(),
            'metrics': results['metrics'],
            'feature_names': feature_names,
            'model_parameters': {
                'n_clusters': n_clusters,
                'linkage': linkage,
                'metric': metric
            }
        }
        
        return format_output(formatted_results, output_format, save_path)
        
    except Exception as e:
        logger.error(f"Hierarchical Clustering failed: {str(e)}")
        return format_output({'error': str(e)}, output_format)


def double_ml_adapter(
    X_data: Optional[Union[List[float], List[List[float]]]] = None,
    y_data: Optional[List[float]] = None,
    d_data: Optional[List[float]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    treatment_type: str = 'continuous',
    n_folds: int = 5,
    random_state: int = 42,
    output_format: str = 'json',
    save_path: Optional[str] = None
) -> str:
    """Double Machine Learning analysis adapter"""
    try:
        if file_path:
            data = DataLoader.load_from_file(file_path)
            if isinstance(data, dict):
                X_data = data.get('X', data.get('features'))
                y_data = data.get('y', data.get('outcome'))
                d_data = data.get('d', data.get('treatment'))
                if feature_names is None:
                    feature_names = data.get('feature_names')
        
        if X_data is None or y_data is None or d_data is None:
            raise ValueError("X_data, y_data, and d_data must be provided")
        
        X = np.array(X_data)
        y = np.array(y_data)
        d = np.array(d_data)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        results = double_ml_analysis(
            X=X, y=y, d=d, treatment_type=treatment_type,
            n_folds=n_folds, random_state=random_state
        )
        
        formatted_results = {
            'model_type': 'double_ml',
            'treatment_type': treatment_type,
            'effect': float(results['effect']),
            'se': float(results['se']),
            'ci': results['ci'],
            'pval': float(results['pval']),
            'feature_names': feature_names,
            'model_parameters': {
                'treatment_type': treatment_type,
                'n_folds': n_folds,
                'random_state': random_state
            }
        }
        
        return format_output(formatted_results, output_format, save_path)
        
    except Exception as e:
        logger.error(f"Double ML failed: {str(e)}")
        return format_output({'error': str(e)}, output_format)


def causal_forest_adapter(
    X_data: Optional[Union[List[float], List[List[float]]]] = None,
    y_data: Optional[List[float]] = None,
    w_data: Optional[List[float]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    n_estimators: int = 100,
    min_samples_leaf: int = 5,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    honest: bool = True,
    output_format: str = 'json',
    save_path: Optional[str] = None
) -> str:
    """Causal Forest analysis adapter"""
    try:
        if file_path:
            data = DataLoader.load_from_file(file_path)
            if isinstance(data, dict):
                X_data = data.get('X', data.get('features'))
                y_data = data.get('y', data.get('outcome'))
                w_data = data.get('w', data.get('treatment'))
                if feature_names is None:
                    feature_names = data.get('feature_names')
        
        if X_data is None or y_data is None or w_data is None:
            raise ValueError("X_data, y_data, and w_data must be provided")
        
        X = np.array(X_data)
        y = np.array(y_data)
        w = np.array(w_data)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        results = causal_forest_analysis(
            X=X, y=y, w=w, n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf, max_depth=max_depth,
            random_state=random_state, honest=honest
        )
        
        te_results = results['treatment_effects']
        formatted_results = {
            'model_type': 'causal_forest',
            'cate': te_results['cate'].tolist(),
            'ate': float(te_results['ate']),
            'cate_se': float(te_results['cate_se']),
            'feature_names': feature_names,
            'model_parameters': {
                'n_estimators': n_estimators,
                'min_samples_leaf': min_samples_leaf,
                'max_depth': max_depth,
                'honest': honest,
                'random_state': random_state
            }
        }
        
        return format_output(formatted_results, output_format, save_path)
        
    except Exception as e:
        logger.error(f"Causal Forest failed: {str(e)}")
        return format_output({'error': str(e)}, output_format)