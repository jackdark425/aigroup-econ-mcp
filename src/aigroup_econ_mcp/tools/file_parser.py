"""
文件解析模块
支持CSV和JSON格式文件的智能解析和数据转换
"""

import json
import csv
from typing import Dict, List, Any, Union, Tuple, Optional
from pathlib import Path
import io
import base64


class FileParser:
    """文件解析器，支持CSV和JSON格式"""
    
    @staticmethod
    def parse_file_content(
        content: str,
        file_format: str = "auto"
    ) -> Dict[str, Any]:
        """
        解析文件内容
        
        Args:
            content: 文件内容（base64编码的字符串或直接文本）
            file_format: 文件格式 ("csv", "json", "auto")
        
        Returns:
            解析后的数据字典，包含：
            - data: 数据内容
            - variables: 变量名列表
            - format: 检测到的格式
            - data_type: 数据类型（'univariate', 'multivariate', 'time_series', 'panel'）
        """
        # 尝试检测是否为base64编码
        try:
            decoded_content = base64.b64decode(content).decode('utf-8')
        except:
            decoded_content = content
        
        # 自动检测格式
        if file_format == "auto":
            file_format = FileParser._detect_format(decoded_content)
        
        if file_format == "csv":
            return FileParser._parse_csv(decoded_content)
        elif file_format == "json":
            return FileParser._parse_json(decoded_content)
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")
    
    @staticmethod
    def _detect_format(content: str) -> str:
        """自动检测文件格式"""
        # 尝试解析JSON
        try:
            json.loads(content.strip())
            return "json"
        except:
            pass
        
        # 检测CSV特征
        if ',' in content or '\t' in content:
            return "csv"
        
        raise ValueError("无法自动检测文件格式，请明确指定")
    
    @staticmethod
    def _parse_csv(content: str) -> Dict[str, Any]:
        """
        解析CSV文件
        
        支持的格式：
        1. 带表头的列数据
        2. 无表头的纯数值数据
        """
        lines = content.strip().split('\n')
        if not lines:
            raise ValueError("CSV文件为空")
        
        # 检测分隔符
        delimiter = FileParser._detect_delimiter(lines[0])
        
        # 使用csv.reader解析
        reader = csv.reader(io.StringIO(content), delimiter=delimiter)
        rows = list(reader)
        
        if not rows:
            raise ValueError("CSV文件没有数据")
        
        # 检测是否有表头
        has_header = FileParser._has_header(rows)
        
        if has_header:
            headers = rows[0]
            data_rows = rows[1:]
        else:
            # 自动生成列名
            headers = [f"var{i+1}" for i in range(len(rows[0]))]
            data_rows = rows
        
        # 转换为数值数据
        parsed_data = {}
        for i, header in enumerate(headers):
            column_data = []
            for row in data_rows:
                if i < len(row):
                    try:
                        # 尝试转换为浮点数
                        value = float(row[i].strip())
                        column_data.append(value)
                    except ValueError:
                        # 如果无法转换，可能是缺失值或字符串
                        pass
            
            if column_data:  # 只保留成功解析的列
                parsed_data[header.strip()] = column_data
        
        if not parsed_data:
            raise ValueError("CSV文件中没有有效的数值数据")
        
        # 检测数据类型
        data_type = FileParser._detect_data_type(parsed_data)
        
        return {
            "data": parsed_data,
            "variables": list(parsed_data.keys()),
            "format": "csv",
            "data_type": data_type,
            "n_variables": len(parsed_data),
            "n_observations": len(next(iter(parsed_data.values())))
        }
    
    @staticmethod
    def _parse_json(content: str) -> Dict[str, Any]:
        """
        解析JSON文件
        
        支持的格式：
        1. {"变量名": [数据列表], ...}
        2. [{"变量1": 值, "变量2": 值, ...}, ...]
        3. {"data": {...}, "metadata": {...}}
        """
        try:
            json_data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON格式错误: {str(e)}")
        
        # 格式1: 直接的变量-数据字典
        if isinstance(json_data, dict) and all(
            isinstance(v, list) for v in json_data.values()
        ):
            # 检查是否所有值都是数值列表
            parsed_data = {}
            for key, values in json_data.items():
                if key.lower() in ['metadata', 'info', 'description']:
                    continue  # 跳过元数据字段
                
                try:
                    numeric_values = [float(v) for v in values]
                    parsed_data[key] = numeric_values
                except (ValueError, TypeError):
                    pass  # 跳过非数值列
            
            if parsed_data:
                data_type = FileParser._detect_data_type(parsed_data)
                return {
                    "data": parsed_data,
                    "variables": list(parsed_data.keys()),
                    "format": "json",
                    "data_type": data_type,
                    "n_variables": len(parsed_data),
                    "n_observations": len(next(iter(parsed_data.values())))
                }
        
        # 格式2: 记录数组格式
        elif isinstance(json_data, list) and json_data and isinstance(json_data[0], dict):
            # 转换为变量-数据字典
            parsed_data = {}
            for record in json_data:
                for key, value in record.items():
                    if key not in parsed_data:
                        parsed_data[key] = []
                    try:
                        parsed_data[key].append(float(value))
                    except (ValueError, TypeError):
                        pass
            
            if parsed_data:
                data_type = FileParser._detect_data_type(parsed_data)
                return {
                    "data": parsed_data,
                    "variables": list(parsed_data.keys()),
                    "format": "json",
                    "data_type": data_type,
                    "n_variables": len(parsed_data),
                    "n_observations": len(next(iter(parsed_data.values())))
                }
        
        # 格式3: 包含data字段的结构
        elif isinstance(json_data, dict) and "data" in json_data:
            return FileParser._parse_json(json.dumps(json_data["data"]))
        
        raise ValueError("不支持的JSON数据格式")
    
    @staticmethod
    def _detect_delimiter(line: str) -> str:
        """检测CSV分隔符"""
        # 常见分隔符
        delimiters = [',', '\t', ';', '|']
        counts = {d: line.count(d) for d in delimiters}
        # 返回出现次数最多的分隔符
        return max(counts.items(), key=lambda x: x[1])[0]
    
    @staticmethod
    def _has_header(rows: List[List[str]]) -> bool:
        """检测CSV是否有表头"""
        if len(rows) < 2:
            return False
        
        # 检查第一行是否包含非数值字符串
        first_row = rows[0]
        
        # 如果第一行有任何元素无法转换为数字，认为有表头
        for cell in first_row:
            try:
                float(cell.strip())
            except ValueError:
                return True
        
        return False
    
    @staticmethod
    def _detect_data_type(data: Dict[str, List[float]]) -> str:
        """
        检测数据类型
        
        Returns:
            - 'univariate': 单变量
            - 'multivariate': 多变量
            - 'time_series': 时间序列（通过变量名推断）
            - 'panel': 面板数据（通过变量名推断）
        """
        n_vars = len(data)
        var_names = [v.lower() for v in data.keys()]
        
        # 检查是否包含时间/日期相关的变量名
        time_keywords = ['time', 'date', 'year', 'month', 'day', 'period', 'quarter']
        has_time_var = any(any(kw in var for kw in time_keywords) for var in var_names)
        
        # 检查是否包含实体/ID相关的变量名
        entity_keywords = ['id', 'entity', 'firm', 'company', 'country', 'region']
        has_entity_var = any(any(kw in var for kw in entity_keywords) for var in var_names)
        
        if n_vars == 1:
            return 'univariate'
        elif has_entity_var and has_time_var:
            return 'panel'
        elif has_time_var or n_vars >= 2:
            return 'time_series'
        else:
            return 'multivariate'
    
    @staticmethod
    def convert_to_tool_format(
        parsed_data: Dict[str, Any],
        tool_type: str
    ) -> Dict[str, Any]:
        """
        将解析后的数据转换为工具所需的格式
        
        Args:
            parsed_data: parse_file_content返回的数据
            tool_type: 工具类型
                - 'single_var': 单变量 (List[float])
                - 'multi_var_dict': 多变量字典 (Dict[str, List[float]])
                - 'multi_var_matrix': 多变量矩阵 (List[List[float]])
                - 'regression': 回归分析 (y_data, x_data)
                - 'panel': 面板数据 (y_data, x_data, entity_ids, time_periods)
        
        Returns:
            转换后的数据字典
        """
        data = parsed_data["data"]
        variables = parsed_data["variables"]
        
        if tool_type == 'single_var':
            # 返回第一个变量的数据
            return {
                "data": data[variables[0]],
                "variable_name": variables[0]
            }
        
        elif tool_type == 'multi_var_dict':
            # 直接返回字典格式
            return {"data": data}
        
        elif tool_type == 'multi_var_matrix':
            # 转换为矩阵格式 (List[List[float]])
            n_obs = len(data[variables[0]])
            matrix = []
            for i in range(n_obs):
                row = [data[var][i] for var in variables]
                matrix.append(row)
            
            return {
                "data": matrix,
                "feature_names": variables
            }
        
        elif tool_type == 'regression':
            # 假设最后一个变量是因变量，其余是自变量
            if len(variables) < 2:
                raise ValueError("回归分析至少需要2个变量（1个因变量和至少1个自变量）")
            
            y_var = variables[-1]
            x_vars = variables[:-1]
            
            y_data = data[y_var]
            n_obs = len(y_data)
            
            # 构建x_data矩阵
            x_data = []
            for i in range(n_obs):
                row = [data[var][i] for var in x_vars]
                x_data.append(row)
            
            return {
                "y_data": y_data,
                "x_data": x_data,
                "feature_names": x_vars,
                "y_variable": y_var
            }
        
        elif tool_type == 'panel':
            # 识别实体ID、时间标识和数据变量
            entity_var = None
            time_var = None
            data_vars = []
            
            entity_keywords = ['id', 'entity', 'firm', 'company', 'country', 'region']
            time_keywords = ['time', 'date', 'year', 'month', 'day', 'period', 'quarter']
            
            for var in variables:
                var_lower = var.lower()
                if any(kw in var_lower for kw in entity_keywords) and entity_var is None:
                    entity_var = var
                elif any(kw in var_lower for kw in time_keywords) and time_var is None:
                    time_var = var
                else:
                    data_vars.append(var)
            
            if not entity_var or not time_var:
                raise ValueError("面板数据需要包含实体ID和时间标识变量")
            
            if len(data_vars) < 2:
                raise ValueError("面板数据至少需要2个数据变量（1个因变量和至少1个自变量）")
            
            # 转换ID和时间为字符串
            entity_ids = [str(int(x)) if x == int(x) else str(x) for x in data[entity_var]]
            time_periods = [str(int(x)) if x == int(x) else str(x) for x in data[time_var]]
            
            # 假设最后一个数据变量是因变量
            y_var = data_vars[-1]
            x_vars = data_vars[:-1]
            
            y_data = data[y_var]
            n_obs = len(y_data)
            
            # 构建x_data矩阵
            x_data = []
            for i in range(n_obs):
                row = [data[var][i] for var in x_vars]
                x_data.append(row)
            
            return {
                "y_data": y_data,
                "x_data": x_data,
                "entity_ids": entity_ids,
                "time_periods": time_periods,
                "feature_names": x_vars,
                "y_variable": y_var
            }
        
        else:
            raise ValueError(f"不支持的工具类型: {tool_type}")
    
    @staticmethod
    def auto_detect_tool_params(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        自动检测并推荐适合的工具参数
        
        Args:
            parsed_data: parse_file_content返回的数据
        
        Returns:
            推荐的工具和参数
        """
        data_type = parsed_data["data_type"]
        n_vars = parsed_data["n_variables"]
        n_obs = parsed_data["n_observations"]
        
        recommendations = {
            "data_type": data_type,
            "suggested_tools": [],
            "warnings": []
        }
        
        # 根据数据类型推荐工具
        if data_type == 'univariate':
            recommendations["suggested_tools"] = [
                "descriptive_statistics",
                "hypothesis_testing",
                "time_series_analysis"
            ]
        elif data_type == 'multivariate':
            recommendations["suggested_tools"] = [
                "descriptive_statistics",
                "correlation_analysis",
                "ols_regression",
                "random_forest_regression_analysis",
                "lasso_regression_analysis"
            ]
        elif data_type == 'time_series':
            recommendations["suggested_tools"] = [
                "time_series_analysis",
                "var_model_analysis",
                "garch_model_analysis"
            ]
        elif data_type == 'panel':
            recommendations["suggested_tools"] = [
                "panel_fixed_effects",
                "panel_random_effects",
                "panel_hausman_test",
                "panel_unit_root_test"
            ]
        
        # 添加警告
        if n_obs < 30:
            recommendations["warnings"].append(
                f"样本量较小（{n_obs}个观测），统计推断可能不可靠"
            )
        
        if n_vars > 10:
            recommendations["warnings"].append(
                f"变量数量较多（{n_vars}个变量），可能需要特征选择"
            )
        
        if n_vars > n_obs / 10:
            recommendations["warnings"].append(
                "变量数量接近样本量的1/10，可能存在过拟合风险"
            )
        
        return recommendations


def parse_file_input(
    file_content: Optional[str] = None,
    file_format: str = "auto"
) -> Optional[Dict[str, Any]]:
    """
    便捷函数：解析文件输入
    
    Args:
        file_content: 文件内容（可选）
        file_format: 文件格式
    
    Returns:
        解析后的数据，如果file_content为None则返回None
    """
    if file_content is None:
        return None
    
    parser = FileParser()
    return parser.parse_file_content(file_content, file_format)