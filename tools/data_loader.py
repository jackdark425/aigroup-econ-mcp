"""

CSVJSONTXT
"""

from typing import Dict, List, Union
from pathlib import Path
import pandas as pd
from .file_parser import FileParser


async def load_data_if_path(
    data: Union[Dict[str, List[float]], str],
    ctx = None
) -> Dict[str, List[float]]:
    """
    
    
    Args:
        data: CSV/JSON/TXT
        ctx: MCP
        
    Returns:
        
        
    Raises:
        ValueError: 
    """
    # 
    if isinstance(data, dict):
        return data
    
    # 
    if isinstance(data, str):
        if ctx:
            await ctx.info(f" : {data}")
        
        try:
            # 
            path = Path(data)
            if not path.exists():
                raise ValueError(f": {data}")
            
            # FileParserCSV/JSON/TXT
            parsed = FileParser.parse_file_path(str(path), "auto")
            
            # 
            result = parsed["data"]
            
            if ctx:
                await ctx.info(
                    f" {parsed['format'].upper()}"
                    f"{parsed['n_variables']}{parsed['n_observations']}"
                )
            
            return result
            
        except FileNotFoundError:
            raise ValueError(f": {data}")
        except Exception as e:
            raise ValueError(f": {str(e)}")
    
    # 
    raise TypeError(f": {type(data)}Dictstr")


async def load_single_var_if_path(
    data: Union[List[float], str],
    ctx = None,
    column_name: str = None
) -> List[float]:
    """
    
    
    Args:
        data: CSV/JSON/TXT
        ctx: MCP
        column_name: 
        
    Returns:
        
        
    Raises:
        ValueError: 
    """
    # 
    if isinstance(data, list):
        return data
    
    # 
    if isinstance(data, str):
        if ctx:
            await ctx.info(f" : {data}")
        
        try:
            # 
            path = Path(data)
            if not path.exists():
                raise ValueError(f": {data}")
            
            # FileParser
            parsed = FileParser.parse_file_path(str(path), "auto")
            data_dict = parsed["data"]
            
            # 
            if column_name:
                if column_name not in data_dict:
                    raise ValueError(
                        f"'{column_name}'"
                        f": {list(data_dict.keys())}"
                    )
                result = data_dict[column_name]
            else:
                # 
                first_col = parsed["variables"][0]
                result = data_dict[first_col]
                if ctx:
                    await ctx.info(f": {first_col}")
            
            if ctx:
                await ctx.info(
                    f" {parsed['format'].upper()}{len(result)}"
                )
            
            return result
            
        except FileNotFoundError:
            raise ValueError(f": {data}")
        except Exception as e:
            raise ValueError(f": {str(e)}")
    
    # 
    raise TypeError(f": {type(data)}Liststr")


async def load_x_data_if_path(
    data: Union[List[List[float]], str],
    ctx = None
) -> List[List[float]]:
    """
    
    
    Args:
        data: CSV/JSON/TXT
        ctx: MCP
        
    Returns:
        
        
    Raises:
        ValueError: 
    """
    # 
    if isinstance(data, list) and all(isinstance(item, list) for item in data):
        return data
    
    # 
    if isinstance(data, str):
        if ctx:
            await ctx.info(f" : {data}")
        
        try:
            # 
            path = Path(data)
            if not path.exists():
                raise ValueError(f": {data}")
            
            # FileParser
            parsed = FileParser.parse_file_path(str(path), "auto")
            data_dict = parsed["data"]
            
            # 
            variables = parsed["variables"]
            n_obs = parsed["n_observations"]
            
            result = []
            for i in range(n_obs):
                row = [data_dict[var][i] for var in variables]
                result.append(row)
            
            if ctx:
                await ctx.info(
                    f" {parsed['format'].upper()}"
                    f"{len(result)}{len(variables)}"
                )
            
            return result
            
        except FileNotFoundError:
            raise ValueError(f": {data}")
        except Exception as e:
            raise ValueError(f": {str(e)}")
    
    # 
    raise TypeError(f": {type(data)}List[List[float]]str")