"""
计量经济学数据资源模块
提供对各种经济数据源的访问资源
"""

from mcp.server.fastmcp import FastMCP


# 定义经济数据资源
def get_world_bank_resource(dataset_id: str) -> str:
    """获取世界银行数据集资源"""
    return f"World Bank dataset: {dataset_id}\nDescription: Economic indicators data from World Bank"


def get_fred_resource(series_id: str) -> str:
    """获取FRED经济数据系列资源"""
    return f"FRED series: {series_id}\nDescription: Federal Reserve Economic Data series"


def get_oecd_resource(dataset_code: str) -> str:
    """获取OECD数据集资源"""
    return f"OECD dataset: {dataset_code}\nDescription: OECD economic and social data"