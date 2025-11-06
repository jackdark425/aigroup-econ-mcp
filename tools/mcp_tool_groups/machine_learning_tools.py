"""
机器学习方法工具组
包含8种机器学习模型的MCP工具
"""

from typing import List, Optional, Union, Dict, Any
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from ..mcp_tools_registry import ToolGroup
from ..machine_learning_adapter import (
    random_forest_adapter,
    gradient_boosting_adapter,
    svm_adapter,
    neural_network_adapter,
    kmeans_clustering_adapter,
    hierarchical_clustering_adapter,
    double_ml_adapter,
    causal_forest_adapter
)


class MachineLearningTools(ToolGroup):
    """机器学习方法工具组"""
    
    name = "MACHINE LEARNING"
    description = "机器学习模型工具"
    version = "1.0.0"
    
    @classmethod
    def get_tools(cls) -> List[Dict[str, Any]]:
        """返回工具列表"""
        return [
            {
                "name": "ml_random_forest",
                "handler": cls.random_forest_tool,
                "description": "Random Forest Analysis (Regression/Classification)"
            },
            {
                "name": "ml_gradient_boosting",
                "handler": cls.gradient_boosting_tool,
                "description": "Gradient Boosting Machine Analysis"
            },
            {
                "name": "ml_support_vector_machine",
                "handler": cls.svm_tool,
                "description": "Support Vector Machine Analysis"
            },
            {
                "name": "ml_neural_network",
                "handler": cls.neural_network_tool,
                "description": "Neural Network (MLP) Analysis"
            },
            {
                "name": "ml_kmeans_clustering",
                "handler": cls.kmeans_tool,
                "description": "K-Means Clustering Analysis"
            },
            {
                "name": "ml_hierarchical_clustering",
                "handler": cls.hierarchical_clustering_tool,
                "description": "Hierarchical Clustering Analysis"
            },
            {
                "name": "ml_double_machine_learning",
                "handler": cls.double_ml_tool,
                "description": "Double/Debiased Machine Learning for Causal Inference"
            },
            {
                "name": "ml_causal_forest",
                "handler": cls.causal_forest_tool,
                "description": "Causal Forest for Heterogeneous Treatment Effects"
            }
        ]
    
    @classmethod
    def get_help_text(cls) -> str:
        """返回帮助文档"""
        return """
机器学习方法工具组 - 8种机器学习模型

监督学习模型:
1. Random Forest - ml_random_forest
   - 随机森林回归/分类
   - 支持特征重要性分析
   
2. Gradient Boosting - ml_gradient_boosting
   - 梯度提升机（支持sklearn/XGBoost）
   - 高性能集成学习
   
3. Support Vector Machine - ml_support_vector_machine
   - 支持向量机回归/分类
   - 多种核函数选择
   
4. Neural Network - ml_neural_network
   - 多层感知器（MLP）
   - 可配置网络结构

无监督学习模型:
5. K-Means Clustering - ml_kmeans_clustering
   - K均值聚类
   - 聚类质量评估
   
6. Hierarchical Clustering - ml_hierarchical_clustering
   - 层次聚类
   - 树状图可视化

因果推断模型:
7. Double Machine Learning - ml_double_machine_learning
   - 双重/去偏机器学习
   - 处理效应估计
   
8. Causal Forest - ml_causal_forest
   - 因果森林
   - 异质性治疗效应估计
"""
    
    @staticmethod
    async def random_forest_tool(
        X_data: Optional[List] = None,
        y_data: Optional[List[float]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        problem_type: str = 'regression',
        test_size: float = 0.2,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        output_format: str = 'json',
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """随机森林分析"""
        try:
            if ctx:
                await ctx.info("Starting Random Forest analysis...")
            
            result = random_forest_adapter(
                X_data=X_data, y_data=y_data, file_path=file_path,
                feature_names=feature_names, problem_type=problem_type,
                test_size=test_size, n_estimators=n_estimators,
                max_depth=max_depth, random_state=random_state,
                output_format=output_format, save_path=save_path
            )
            
            if ctx:
                await ctx.info("Random Forest analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def gradient_boosting_tool(
        X_data: Optional[List] = None,
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
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """梯度提升分析"""
        try:
            if ctx:
                await ctx.info("Starting Gradient Boosting analysis...")
            
            result = gradient_boosting_adapter(
                X_data=X_data, y_data=y_data, file_path=file_path,
                feature_names=feature_names, algorithm=algorithm,
                problem_type=problem_type, test_size=test_size,
                n_estimators=n_estimators, learning_rate=learning_rate,
                max_depth=max_depth, random_state=random_state,
                output_format=output_format, save_path=save_path
            )
            
            if ctx:
                await ctx.info("Gradient Boosting analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def svm_tool(
        X_data: Optional[List] = None,
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
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """支持向量机分析"""
        try:
            if ctx:
                await ctx.info("Starting SVM analysis...")
            
            result = svm_adapter(
                X_data=X_data, y_data=y_data, file_path=file_path,
                feature_names=feature_names, problem_type=problem_type,
                kernel=kernel, test_size=test_size, C=C, gamma=gamma,
                random_state=random_state, output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("SVM analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def neural_network_tool(
        X_data: Optional[List] = None,
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
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """神经网络分析"""
        try:
            if ctx:
                await ctx.info("Starting Neural Network analysis...")
            
            result = neural_network_adapter(
                X_data=X_data, y_data=y_data, file_path=file_path,
                feature_names=feature_names, problem_type=problem_type,
                hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                solver=solver, test_size=test_size, alpha=alpha,
                learning_rate=learning_rate, learning_rate_init=learning_rate_init,
                max_iter=max_iter, random_state=random_state,
                output_format=output_format, save_path=save_path
            )
            
            if ctx:
                await ctx.info("Neural Network analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def kmeans_tool(
        X_data: Optional[List] = None,
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
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """K均值聚类分析"""
        try:
            if ctx:
                await ctx.info("Starting K-Means Clustering analysis...")
            
            result = kmeans_clustering_adapter(
                X_data=X_data, file_path=file_path, feature_names=feature_names,
                n_clusters=n_clusters, init=init, n_init=n_init,
                max_iter=max_iter, random_state=random_state,
                algorithm=algorithm, use_minibatch=use_minibatch,
                batch_size=batch_size, output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("K-Means Clustering analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def hierarchical_clustering_tool(
        X_data: Optional[List] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        n_clusters: int = 2,
        linkage: str = 'ward',
        metric: str = 'euclidean',
        output_format: str = 'json',
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """层次聚类分析"""
        try:
            if ctx:
                await ctx.info("Starting Hierarchical Clustering analysis...")
            
            result = hierarchical_clustering_adapter(
                X_data=X_data, file_path=file_path, feature_names=feature_names,
                n_clusters=n_clusters, linkage=linkage, metric=metric,
                output_format=output_format, save_path=save_path
            )
            
            if ctx:
                await ctx.info("Hierarchical Clustering analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def double_ml_tool(
        X_data: Optional[List] = None,
        y_data: Optional[List[float]] = None,
        d_data: Optional[List[float]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        treatment_type: str = 'continuous',
        n_folds: int = 5,
        random_state: int = 42,
        output_format: str = 'json',
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """双重机器学习分析"""
        try:
            if ctx:
                await ctx.info("Starting Double Machine Learning analysis...")
            
            result = double_ml_adapter(
                X_data=X_data, y_data=y_data, d_data=d_data,
                file_path=file_path, feature_names=feature_names,
                treatment_type=treatment_type, n_folds=n_folds,
                random_state=random_state, output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Double Machine Learning analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def causal_forest_tool(
        X_data: Optional[List] = None,
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
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """因果森林分析"""
        try:
            if ctx:
                await ctx.info("Starting Causal Forest analysis...")
            
            result = causal_forest_adapter(
                X_data=X_data, y_data=y_data, w_data=w_data,
                file_path=file_path, feature_names=feature_names,
                n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                max_depth=max_depth, random_state=random_state,
                honest=honest, output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Causal Forest analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise