"""


"""

import json
import csv
from typing import Dict, List, Any, Union, Tuple, Optional, Callable
from pathlib import Path
from functools import wraps
import io
import base64


class FileParser:
    """CSVJSONTXT"""
    
    @staticmethod
    def parse_file_path(
        file_path: str,
        file_format: str = "auto"
    ) -> Dict[str, Any]:
        """
        
        
        Args:
            file_path: 
            file_format:  ("csv", "json", "auto")
        
        Returns:
            
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f": {file_path}")
        
        if not path.is_file():
            raise ValueError(f": {file_path}")
        
        # 
        if file_format == "auto":
            ext = path.suffix.lower()
            if ext == '.csv':
                file_format = "csv"
            elif ext in ['.json', '.jsonl']:
                file_format = "json"
            elif ext == '.txt':
                file_format = 'txt'
            else:
                # 
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return FileParser.parse_file_content(content, "auto")
        
        # 
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return FileParser.parse_file_content(content, file_format)
    
    @staticmethod
    def parse_file_content(
        content: str,
        file_format: str = "auto"
    ) -> Dict[str, Any]:
        """
        
        
        Args:
            content: base64
            file_format:  ("csv", "json", "auto")
        
        Returns:
            
            - data: 
            - variables: 
            - format: 
            - data_type: 'univariate', 'multivariate', 'time_series', 'panel'
        """
        # base64
        try:
            decoded_content = base64.b64decode(content).decode('utf-8')
        except:
            decoded_content = content
        
        # 
        if file_format == "auto":
            file_format = FileParser._detect_format(decoded_content)
        
        if file_format == "csv":
            return FileParser._parse_csv(decoded_content)
        elif file_format == "txt":
            return FileParser._parse_txt(decoded_content)
        elif file_format == "json":
            return FileParser._parse_json(decoded_content)
        else:
            raise ValueError(f": {file_format}")
    
    @staticmethod
    def _detect_format(content: str) -> str:
        """"""
        # JSON
        try:
            json.loads(content.strip())
            return "json"
        except:
            pass
        
        # CSV

        # TXT
        lines = content.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # 
            if ':' in first_line or '=' in first_line:
                return "txt"
            
            # 
            try:
                float(first_line)
                return "txt"
            except ValueError:
                # 
                parts = first_line.split()
                if parts:
                    try:
                        for part in parts:
                            float(part)
                        return "txt"
                    except ValueError:
                        pass
        
        if ',' in content or '\t' in content:
            return "csv"
        
        raise ValueError("")
    
    @staticmethod
    def _parse_csv(content: str) -> Dict[str, Any]:
        """
        CSV
        
        
        1. 
        2. 
        """
        lines = content.strip().split('\n')
        if not lines:
            raise ValueError("CSV")
        
        # 
        delimiter = FileParser._detect_delimiter(lines[0])
        
        # csv.reader
        reader = csv.reader(io.StringIO(content), delimiter=delimiter)
        rows = list(reader)
        
        if not rows:
            raise ValueError("CSV")
        
        # 
        has_header = FileParser._has_header(rows)
        
        if has_header:
            headers = rows[0]
            data_rows = rows[1:]
        else:
            # 
            headers = [f"var{i+1}" for i in range(len(rows[0]))]
            data_rows = rows
        
        # 
        parsed_data = {}
        for i, header in enumerate(headers):
            column_data = []
            for row in data_rows:
                if i < len(row):
                    try:
                        # 
                        value = float(row[i].strip())
                        column_data.append(value)
                    except ValueError:
                        # ID
                        column_data.append(row[i].strip())
            
            if column_data:  # 
                parsed_data[header.strip()] = column_data
        
        if not parsed_data:
            raise ValueError("CSV")
        
        # 
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
        JSON
        
        
        1. {"": [], ...}
        2. [{"1": , "2": , ...}, ...]
        3. {"data": {...}, "metadata": {...}}
        """
        try:
            json_data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON: {str(e)}")
        
        # 1: -
        if isinstance(json_data, dict) and all(
            isinstance(v, list) for v in json_data.values()
        ):
            # ID
            parsed_data = {}
            for key, values in json_data.items():
                if key.lower() in ['metadata', 'info', 'description']:
                    continue  # 
                
                # 
                converted_values = []
                for v in values:
                    try:
                        # 
                        converted_values.append(float(v))
                    except (ValueError, TypeError):
                        # 
                        converted_values.append(v)
                
                parsed_data[key] = converted_values
            
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
        
        # 2: 
        elif isinstance(json_data, list) and json_data and isinstance(json_data[0], dict):
            # -
            parsed_data = {}
            for record in json_data:
                for key, value in record.items():
                    if key not in parsed_data:
                        parsed_data[key] = []
                    # 
                    try:
                        parsed_data[key].append(float(value))
                    except (ValueError, TypeError):
                        # 
                        parsed_data[key].append(value)
            
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
        
        # 3: data
        elif isinstance(json_data, dict) and "data" in json_data:
            return FileParser._parse_json(json.dumps(json_data["data"]))
        
        raise ValueError("JSON")

    
    @staticmethod
    def _parse_txt(content: str) -> Dict[str, Any]:
        """
        TXT
        
        
        1. 
        2. /
        3. :    
        """
        lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
        if not lines:
            raise ValueError("TXT")
        
        # 
        first_line = lines[0]
        
        # 1: 
        if ':' in first_line or '=' in first_line:
            return FileParser._parse_txt_keyvalue(lines)
        
        # 2: 
        if ' ' in first_line or '\t' in first_line:
            return FileParser._parse_txt_multicolumn(lines)
        
        # 3: 
        return FileParser._parse_txt_single_column(lines)
    
    @staticmethod
    def _parse_txt_single_column(lines: List[str]) -> Dict[str, Any]:
        """TXT"""
        data_list = []
        for i, line in enumerate(lines, 1):
            try:
                value = float(line)
                data_list.append(value)
            except ValueError:
                raise ValueError(f"{i}: {line}")
        
        if not data_list:
            raise ValueError("TXT")
        
        parsed_data = {"data": data_list}
        
        return {
            "data": parsed_data,
            "variables": ["data"],
            "format": "txt",
            "data_type": "univariate",
            "n_variables": 1,
            "n_observations": len(data_list)
        }
    
    @staticmethod
    def _parse_txt_multicolumn(lines: List[str]) -> Dict[str, Any]:
        """TXT"""
        # 
        first_line = lines[0]
        if '\t' in first_line:
            delimiter = '\t'
        else:
            delimiter = None  # split()
        
        # 
        all_rows = []
        for line in lines:
            if delimiter:
                parts = line.split(delimiter)
            else:
                parts = line.split()
            all_rows.append([p.strip() for p in parts if p.strip()])
        
        if not all_rows:
            raise ValueError("TXT")
        
        # 
        has_header = False
        first_row = all_rows[0]
        for cell in first_row:
            try:
                float(cell)
            except ValueError:
                has_header = True
                break
        
        if has_header:
            headers = first_row
            data_rows = all_rows[1:]
        else:
            n_cols = len(first_row)
            headers = [f"var{i+1}" for i in range(n_cols)]
            data_rows = all_rows
        
        # 
        parsed_data = {}
        for i, header in enumerate(headers):
            column_data = []
            for row in data_rows:
                if i < len(row):
                    try:
                        value = float(row[i])
                        column_data.append(value)
                    except ValueError:
                        column_data.append(row[i])
            
            if column_data:
                parsed_data[header] = column_data
        
        if not parsed_data:
            raise ValueError("TXT")
        
        data_type = FileParser._detect_data_type(parsed_data)
        
        return {
            "data": parsed_data,
            "variables": list(parsed_data.keys()),
            "format": "txt",
            "data_type": data_type,
            "n_variables": len(parsed_data),
            "n_observations": len(next(iter(parsed_data.values())))
        }
    
    @staticmethod
    def _parse_txt_keyvalue(lines: List[str]) -> Dict[str, Any]:
        """TXT"""
        parsed_data = {}
        
        for line in lines:
            if ':' in line:
                separator = ':'
            elif '=' in line:
                separator = '='
            else:
                continue
            
            parts = line.split(separator, 1)
            if len(parts) != 2:
                continue
            
            var_name = parts[0].strip()
            value_str = parts[1].strip()
            values = value_str.split()
            
            try:
                if len(values) == 1:
                    parsed_data[var_name] = [float(values[0])]
                else:
                    parsed_data[var_name] = [float(v) for v in values]
            except ValueError:
                if len(values) == 1:
                    parsed_data[var_name] = [values[0]]
                else:
                    parsed_data[var_name] = values
        
        if not parsed_data:
            raise ValueError("TXT")
        
        data_type = FileParser._detect_data_type(parsed_data)
        n_observations = max(len(v) for v in parsed_data.values())
        
        return {
            "data": parsed_data,
            "variables": list(parsed_data.keys()),
            "format": "txt",
            "data_type": data_type,
            "n_variables": len(parsed_data),
            "n_observations": n_observations
        }
    
    @staticmethod
    def _detect_delimiter(line: str) -> str:
        """CSV"""
        # 
        delimiters = [',', '\t', ';', '|']
        counts = {d: line.count(d) for d in delimiters}
        # 
        return max(counts.items(), key=lambda x: x[1])[0]
    
    @staticmethod
    def _has_header(rows: List[List[str]]) -> bool:
        """CSV"""
        if len(rows) < 2:
            return False
        
        # 
        first_row = rows[0]
        
        # 
        for cell in first_row:
            try:
                float(cell.strip())
            except ValueError:
                return True
        
        return False
    
    @staticmethod
    def _detect_data_type(data: Dict[str, List]) -> str:
        """
        
        
        Returns:
            - 'univariate': 
            - 'multivariate': 
            - 'time_series': 
            - 'panel': 
        """
        n_vars = len(data)
        var_names = [v.lower() for v in data.keys()]
        
        # /
        time_keywords = ['time', 'date', 'year', 'month', 'day', 'period', 'quarter']
        has_time_var = any(any(kw in var for kw in time_keywords) for var in var_names)
        
        # /ID
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
        
        
        Args:
            parsed_data: parse_file_content
            tool_type: 
                - 'single_var':  (List[float])
                - 'multi_var_dict':  (Dict[str, List[float]])
                - 'multi_var_matrix':  (List[List[float]])
                - 'regression':  (y_data, x_data)
                - 'panel':  (y_data, x_data, entity_ids, time_periods)
        
        Returns:
            
        """
        data = parsed_data["data"]
        variables = parsed_data["variables"]
        
        if tool_type == 'single_var':
            #
            var_name = variables[0]
            var_data = data[var_name]
            return {
                "data": var_data,
                "variable_name": var_name
            }
        
        elif tool_type == 'multi_var_dict':
            # 
            return {"data": data}
        
        elif tool_type == 'time_series':
            # multi_var_dict
            return {"data": data}
        
        elif tool_type == 'multi_var_matrix':
            #  (List[List[float]])
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
            # 
            if len(variables) < 2:
                raise ValueError("211")
            
            y_var = variables[-1]
            x_vars = variables[:-1]
            
            y_data = data[y_var]
            n_obs = len(y_data)
            
            # x_data
            x_data = []
            for i in range(n_obs):
                row = [data[var][i] for var in x_vars]
                x_data.append(row)
            
            return {
                "y_data": y_data,
                "y_variable": y_var,
                "x_data": x_data,
                "feature_names": x_vars
            }
        
        elif tool_type == 'panel':
            # ID
            entity_var = None
            time_var = None
            data_vars = []
            
            entity_keywords = ['id', 'entity', 'firm', 'company', 'country', 'region']
            time_keywords = ['time', 'date', 'year', 'month', 'day', 'period', 'quarter']
            
            # 
            print(f"Debug: ...")
            for var in variables:
                var_lower = var.lower()
                print(f"Debug:  '{var}' (: '{var_lower}')")
                
                # ID
                is_entity = any(kw in var_lower for kw in entity_keywords)
                is_time = any(kw in var_lower for kw in time_keywords)
                
                if is_entity and entity_var is None:
                    entity_var = var
                    print(f"Debug: ID: {var}")
                elif is_time and time_var is None:
                    time_var = var
                    print(f"Debug: : {var}")
                else:
                    data_vars.append(var)
                    print(f"Debug: : {var}")
            
            print(f"Debug: entity_var={entity_var}, time_var={time_var}, data_vars={data_vars}")
            
            if not entity_var or not time_var:
                # 
                available_vars = ', '.join(variables)
                error_msg = f"ID\n"
                error_msg += f": {available_vars}\n"
                error_msg += f"ID: {entity_var if entity_var else ''}\n"
                error_msg += f": {time_var if time_var else ''}\n"
                error_msg += f"ID: {entity_keywords}\n"
                error_msg += f": {time_keywords}"
                raise ValueError(error_msg)
            
            if len(data_vars) < 1:
                raise ValueError(f"1: {data_vars}")
            
            # ID
            entity_ids = [str(x) for x in data[entity_var]]
            time_periods = [str(int(x)) if isinstance(x, float) and x == int(x) else str(x) for x in data[time_var]]
            
            print(f"Debug: entity_ids: {entity_ids[:5]}")
            print(f"Debug: time_periods: {time_periods[:5]}")
            
            # 
            if len(data_vars) == 1:
                y_var = data_vars[0]
                y_data = data[y_var]
                # 
                n_obs = len(y_data)
                x_data = [[1.0] for _ in range(n_obs)]
                x_vars = ['const']
            else:
                # 
                y_var = data_vars[-1]
                x_vars = data_vars[:-1]
                
                y_data = data[y_var]
                n_obs = len(y_data)
                
                # x_data
                x_data = []
                for i in range(n_obs):
                    row = [data[var][i] for var in x_vars]
                    x_data.append(row)
            
            return {
                "y_data": y_data,
                "y_variable": y_var,
                "x_data": x_data,
                "entity_ids": entity_ids,
                "time_periods": time_periods,
                "feature_names": x_vars
            }
        
        else:
            raise ValueError(f": {tool_type}")
    
    @staticmethod
    def auto_detect_tool_params(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        
        
        Args:
            parsed_data: parse_file_content
        
        Returns:
            
        """
        data_type = parsed_data["data_type"]
        n_vars = parsed_data["n_variables"]
        n_obs = parsed_data["n_observations"]
        
        recommendations = {
            "data_type": data_type,
            "suggested_tools": [],
            "warnings": []
        }
        
        # 
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
        
        # 
        if n_obs < 30:
            recommendations["warnings"].append(
                f"{n_obs}"
            )
        
        if n_vars > 10:
            recommendations["warnings"].append(
                f"{n_vars}"
            )
        
        if n_vars > n_obs / 10:
            recommendations["warnings"].append(
                "1/10"
            )
        
        return recommendations


# ============================================================================
# 
# ============================================================================

class FileInputHandler:
    """
    
    
    
    """
    
    @staticmethod
    def process_input(
        file_content: Optional[str],
        file_format: str,
        tool_type: str,
        data_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        
        
        Args:
            file_content: 
            file_format: 
            tool_type: 
            data_params: 
        
        Returns:
            
        """
        # 
        if file_content is None:
            return data_params
        
        # 
        parsed = FileParser.parse_file_content(file_content, file_format)
        
        # 
        converted = FileParser.convert_to_tool_format(parsed, tool_type)
        
        # 
        result = {**data_params, **converted}
        
        return result
    
    @staticmethod
    def with_file_support(tool_type: str):
        """
        
        
        Args:
            tool_type: single_var, multi_var_dict, regression, panel
        
        Returns:
            
        
        
            @FileInputHandler.with_file_support('regression')
            async def my_regression_tool(y_data, x_data, file_content=None, file_format='auto'):
                # file_contenty_datax_data
                pass
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 
                file_content = kwargs.get('file_content')
                file_format = kwargs.get('file_format', 'auto')
                
                if file_content is not None:
                    # 
                    processed = FileInputHandler.process_input(
                        file_content=file_content,
                        file_format=file_format,
                        tool_type=tool_type,
                        data_params=kwargs
                    )
                    
                    # kwargs
                    kwargs.update(processed)
                
                # 
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator


class FileInputMixin:
    """
    
    
    
    """
    
    def parse_file_input(
        self,
        file_content: Optional[str],
        file_format: str = "auto"
    ) -> Optional[Dict[str, Any]]:
        """"""
        if file_content is None:
            return None
        return FileParser.parse_file_content(file_content, file_format)
    
    def convert_for_tool(
        self,
        parsed_data: Dict[str, Any],
        tool_type: str
    ) -> Dict[str, Any]:
        """"""
        return FileParser.convert_to_tool_format(parsed_data, tool_type)
    
    def get_recommendations(
        self,
        parsed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """"""
        return FileParser.auto_detect_tool_params(parsed_data)


class UnifiedFileInput:
    """
    
    
    
    """
    
    @staticmethod
    async def handle(
        ctx: Any,
        file_content: Optional[str],
        file_format: str,
        tool_type: str,
        original_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        
        
        Args:
            ctx: MCP
            file_content: 
            file_format: 
            tool_type: 
            original_params: 
        
        Returns:
            
        """
        if file_content is None:
            return original_params
        
        try:
            # 
            await ctx.info("...")
            
            # 
            parsed = FileParser.parse_file_content(file_content, file_format)
            
            # 
            await ctx.info(
                f"{parsed['n_variables']}"
                f"{parsed['n_observations']}"
                f"={parsed['data_type']}"
            )
            
            # 
            converted = FileParser.convert_to_tool_format(parsed, tool_type)
            
            # 
            result = {**original_params}
            result.update(converted)
            
            # 
            if tool_type == 'regression':
                await ctx.info(
                    f"={converted.get('y_variable')}"
                    f"={converted.get('feature_names')}"
                )
            elif tool_type == 'panel':
                await ctx.info(
                    f"{len(set(converted.get('entity_ids', [])))}"
                    f"{len(set(converted.get('time_periods', [])))}"
                )
            else:
                await ctx.info(f"{tool_type}")
            
            return result
            
        except Exception as e:
            await ctx.error(f": {str(e)}")
            raise ValueError(f": {str(e)}")


# ============================================================================
# 
# ============================================================================

def parse_file_input(
    file_content: Optional[str] = None,
    file_format: str = "auto"
) -> Optional[Dict[str, Any]]:
    """
    
    
    Args:
        file_content: 
        file_format: 
    
    Returns:
        file_contentNoneNone
    """
    if file_content is None:
        return None
    
    return FileParser.parse_file_content(file_content, file_format)


async def process_file_for_tool(
    ctx: Any,
    file_content: Optional[str],
    file_format: str,
    tool_type: str,
    **kwargs
) -> Dict[str, Any]:
    """
    
    
    
        params = await process_file_for_tool(
            ctx=ctx,
            file_
content=file_content,
            file_format=file_format,
            tool_type='regression',
            y_data=y_data,
            x_data=x_data,
            feature_names=feature_names
        )
        # params 
    """
    return await UnifiedFileInput.handle(
        ctx=ctx,
        file_content=file_content,
        file_format=file_format,
        tool_type=tool_type,
        original_params=kwargs
    )


def create_file_params(
    description: str = "CSVJSON"
) -> Dict[str, Any]:
    """
    
    
    Args:
        description: 
    
    Returns:
        Field()
    """
    return {
        "file_content": {
            "default": None,
            "description": f"""{description}
            
 
- CSV: 
- JSON: {{"": [], ...}}  [{{"1": , ...}}, ...]

 
- base64
- 
- file_content"""
        },
        "file_format": {
            "default": "auto",
            "description": """
            

- "auto": 
- "csv": CSV
- "json": JSON"""
        }
    }