"""
数据加载组件 - 支持多种文件格式
支持txt、json、csv、excel文件的读取和解析
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union
import pandas as pd


class DataLoader:
    """数据加载器，支持多种文件格式"""
    
    @staticmethod
    def load_from_file(file_path: str) -> Dict[str, Any]:
        """
        从文件加载数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            包含y_data和x_data的字典
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式或数据格式错误
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.txt':
            return DataLoader._load_txt(path)
        elif suffix == '.json':
            return DataLoader._load_json(path)
        elif suffix == '.csv':
            return DataLoader._load_csv(path)
        elif suffix in ['.xlsx', '.xls']:
            return DataLoader._load_excel(path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    @staticmethod
    def _load_txt(path: Path) -> Dict[str, Any]:
        """加载txt文件（空格或制表符分隔）"""
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 跳过空行和注释行
        data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        
        if not data_lines:
            raise ValueError("txt文件为空或没有有效数据")
        
        # 解析数据
        data = []
        for line in data_lines:
            # 支持空格和制表符分隔
            row = [float(x) for x in line.split()]
            data.append(row)
        
        return DataLoader._parse_data_matrix(data)
    
    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        """加载json文件"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 支持两种格式：
        # 1. {"y_data": [...], "x_data": [[...], ...]}
        # 2. {"data": [[y, x1, x2, ...], ...]}
        
        if "y_data" in data and "x_data" in data:
            return {
                "y_data": data["y_data"],
                "x_data": data["x_data"],
                "feature_names": data.get("feature_names"),
            }
        elif "data" in data:
            return DataLoader._parse_data_matrix(data["data"])
        else:
            raise ValueError("JSON格式错误：需要包含'y_data'和'x_data'或'data'字段")
    
    @staticmethod
    def _load_csv(path: Path) -> Dict[str, Any]:
        """加载csv文件"""
        df = pd.read_csv(path)
        return DataLoader._parse_dataframe(df)
    
    @staticmethod
    def _load_excel(path: Path) -> Dict[str, Any]:
        """加载excel文件"""
        df = pd.read_excel(path)
        return DataLoader._parse_dataframe(df)
    
    @staticmethod
    def _parse_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """解析DataFrame"""
        if df.empty:
            raise ValueError("数据框为空")
        
        # 第一列为y，其余列为x
        y_data = df.iloc[:, 0].tolist()
        
        if df.shape[1] > 1:
            x_data = df.iloc[:, 1:].values.tolist()
            feature_names = df.columns[1:].tolist()
        else:
            raise ValueError("数据至少需要包含因变量和一个自变量")
        
        return {
            "y_data": y_data,
            "x_data": x_data,
            "feature_names": feature_names,
        }
    
    @staticmethod
    def _parse_data_matrix(data: List[List[float]]) -> Dict[str, Any]:
        """解析数据矩阵（第一列为y，其余列为x）"""
        if not data:
            raise ValueError("数据矩阵为空")
        
        y_data = [row[0] for row in data]
        
        if len(data[0]) > 1:
            x_data = [row[1:] for row in data]
            feature_names = [f"X{i+1}" for i in range(len(data[0]) - 1)]
        else:
            raise ValueError("数据至少需要包含因变量和一个自变量")
        
        return {
            "y_data": y_data,
            "x_data": x_data,
            "feature_names": feature_names,
        }


class MLEDataLoader:
    """MLE专用数据加载器"""
    
    @staticmethod
    def load_from_file(file_path: str) -> Dict[str, Any]:
        """
        从文件加载MLE数据（单列数据）
        
        Args:
            file_path: 文件路径
            
        Returns:
            包含data的字典
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.txt':
            return MLEDataLoader._load_txt(path)
        elif suffix == '.json':
            return MLEDataLoader._load_json(path)
        elif suffix == '.csv':
            return MLEDataLoader._load_csv(path)
        elif suffix in ['.xlsx', '.xls']:
            return MLEDataLoader._load_excel(path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    @staticmethod
    def _load_txt(path: Path) -> Dict[str, Any]:
        """加载txt文件"""
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                data.append(float(line.split()[0]))
        
        return {"data": data}
    
    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        """加载json文件"""
        with open(path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        if isinstance(loaded, dict) and "data" in loaded:
            return {"data": loaded["data"]}
        elif isinstance(loaded, list):
            return {"data": loaded}
        else:
            raise ValueError("JSON格式错误")
    
    @staticmethod
    def _load_csv(path: Path) -> Dict[str, Any]:
        """加载csv文件"""
        df = pd.read_csv(path)
        return {"data": df.iloc[:, 0].tolist()}
    
    @staticmethod
    def _load_excel(path: Path) -> Dict[str, Any]:
        """加载excel文件"""
        df = pd.read_excel(path)
        return {"data": df.iloc[:, 0].tolist()}