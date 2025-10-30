"""
文件输入功能测试
"""

import pytest
import json
from aigroup_econ_mcp.tools.file_parser import FileParser


class TestFileParser:
    """测试文件解析器"""
    
    def test_parse_csv_with_header(self):
        """测试带表头的CSV解析"""
        csv_content = """GDP,Inflation,Unemployment
3.2,2.1,4.5
2.8,2.3,4.2
3.5,1.9,4.0"""
        
        result = FileParser.parse_file_content(csv_content, "csv")
        
        assert result["format"] == "csv"
        assert result["n_variables"] == 3
        assert result["n_observations"] == 3
        assert "GDP" in result["variables"]
        assert len(result["data"]["GDP"]) == 3
        assert result["data"]["GDP"][0] == 3.2
    
    def test_parse_csv_without_header(self):
        """测试无表头的CSV解析"""
        csv_content = """3.2,2.1,4.5
2.8,2.3,4.2
3.5,1.9,4.0"""
        
        result = FileParser.parse_file_content(csv_content, "csv")
        
        assert result["format"] == "csv"
        assert result["n_variables"] == 3
        assert "var1" in result["variables"]
        assert "var2" in result["variables"]
    
    def test_parse_json_dict_format(self):
        """测试JSON字典格式"""
        json_content = {
            "GDP": [3.2, 2.8, 3.5],
            "Inflation": [2.1, 2.3, 1.9]
        }
        
        result = FileParser.parse_file_content(
            json.dumps(json_content), "json"
        )
        
        assert result["format"] == "json"
        assert result["n_variables"] == 2
        assert len(result["data"]["GDP"]) == 3
    
    def test_parse_json_array_format(self):
        """测试JSON数组格式"""
        json_content = [
            {"GDP": 3.2, "Inflation": 2.1},
            {"GDP": 2.8, "Inflation": 2.3},
            {"GDP": 3.5, "Inflation": 1.9}
        ]
        
        result = FileParser.parse_file_content(
            json.dumps(json_content), "json"
        )
        
        assert result["format"] == "json"
        assert result["n_variables"] == 2
        assert len(result["data"]["GDP"]) == 3
    
    def test_auto_detect_csv(self):
        """测试自动检测CSV格式"""
        csv_content = """var1,var2
1.0,2.0
3.0,4.0"""
        
        result = FileParser.parse_file_content(csv_content, "auto")
        assert result["format"] == "csv"
    
    def test_auto_detect_json(self):
        """测试自动检测JSON格式"""
        json_content = '{"var1": [1.0, 2.0], "var2": [3.0, 4.0]}'
        
        result = FileParser.parse_file_content(json_content, "auto")
        assert result["format"] == "json"
    
    def test_detect_data_type_univariate(self):
        """测试单变量数据类型检测"""
        data = {"var1": [1.0, 2.0, 3.0]}
        result = FileParser._detect_data_type(data)
        assert result == "univariate"
    
    def test_detect_data_type_time_series(self):
        """测试时间序列数据类型检测"""
        data = {
            "date": [1, 2, 3],
            "value": [1.0, 2.0, 3.0]
        }
        result = FileParser._detect_data_type(data)
        assert result == "time_series"
    
    def test_detect_data_type_panel(self):
        """测试面板数据类型检测"""
        data = {
            "company_id": [1, 1, 2, 2],
            "year": [2020, 2021, 2020, 2021],
            "value": [1.0, 2.0, 3.0, 4.0]
        }
        result = FileParser._detect_data_type(data)
        assert result == "panel"
    
    def test_convert_to_single_var(self):
        """测试转换为单变量格式"""
        parsed = {
            "data": {"var1": [1.0, 2.0, 3.0]},
            "variables": ["var1"]
        }
        
        result = FileParser.convert_to_tool_format(parsed, "single_var")
        assert result["data"] == [1.0, 2.0, 3.0]
        assert result["variable_name"] == "var1"
    
    def test_convert_to_multi_var_dict(self):
        """测试转换为多变量字典格式"""
        parsed = {
            "data": {
                "var1": [1.0, 2.0],
                "var2": [3.0, 4.0]
            },
            "variables": ["var1", "var2"]
        }
        
        result = FileParser.convert_to_tool_format(parsed, "multi_var_dict")
        assert "var1" in result["data"]
        assert "var2" in result["data"]
    
    def test_convert_to_regression_format(self):
        """测试转换为回归格式"""
        parsed = {
            "data": {
                "x1": [1.0, 2.0, 3.0],
                "x2": [4.0, 5.0, 6.0],
                "y": [7.0, 8.0, 9.0]
            },
            "variables": ["x1", "x2", "y"]
        }
        
        result = FileParser.convert_to_tool_format(parsed, "regression")
        
        assert len(result["y_data"]) == 3
        assert result["y_variable"] == "y"
        assert len(result["x_data"]) == 3
        assert len(result["x_data"][0]) == 2
        assert result["feature_names"] == ["x1", "x2"]
    
    def test_convert_to_panel_format(self):
        """测试转换为面板数据格式"""
        parsed = {
            "data": {
                "id": [1, 1, 2, 2],
                "time": [2020, 2021, 2020, 2021],
                "x1": [1.0, 2.0, 3.0, 4.0],
                "y": [5.0, 6.0, 7.0, 8.0]
            },
            "variables": ["id", "time", "x1", "y"]
        }
        
        result = FileParser.convert_to_tool_format(parsed, "panel")
        
        assert len(result["entity_ids"]) == 4
        assert len(result["time_periods"]) == 4
        assert len(result["y_data"]) == 4
        assert len(result["x_data"]) == 4
        assert result["feature_names"] == ["x1"]
    
    def test_auto_detect_recommendations(self):
        """测试自动推荐功能"""
        parsed = {
            "data": {"var1": [1.0, 2.0, 3.0]},
            "variables": ["var1"],
            "data_type": "univariate",
            "n_variables": 1,
            "n_observations": 3
        }
        
        result = FileParser.auto_detect_tool_params(parsed)
        
        assert result["data_type"] == "univariate"
        assert "suggested_tools" in result
        assert len(result["suggested_tools"]) > 0
    
    def test_delimiter_detection(self):
        """测试分隔符检测"""
        # 逗号分隔
        line1 = "a,b,c"
        assert FileParser._detect_delimiter(line1) == ","
        
        # 制表符分隔
        line2 = "a\tb\tc"
        assert FileParser._detect_delimiter(line2) == "\t"
        
        # 分号分隔
        line3 = "a;b;c"
        assert FileParser._detect_delimiter(line3) == ";"
    
    def test_header_detection(self):
        """测试表头检测"""
        # 有表头
        rows1 = [["Name", "Age"], ["John", "25"]]
        assert FileParser._has_header(rows1) == True
        
        # 无表头
        rows2 = [["1.0", "2.0"], ["3.0", "4.0"]]
        assert FileParser._has_header(rows2) == False


class TestCSVExamples:
    """测试实际CSV示例"""
    
    def test_regression_data(self):
        """测试回归分析数据"""
        csv_content = """advertising,price,competitors,sales
800,99,3,12000
900,95,3,13500
750,102,4,11800
1000,98,3,14200"""
        
        parsed = FileParser.parse_file_content(csv_content, "csv")
        converted = FileParser.convert_to_tool_format(parsed, "regression")
        
        assert converted["y_variable"] == "sales"
        assert set(converted["feature_names"]) == {"advertising", "price", "competitors"}
        assert len(converted["y_data"]) == 4
        assert len(converted["x_data"]) == 4
    
    def test_panel_data(self):
        """测试面板数据"""
        csv_content = """company_id,year,revenue,employees,profit
1,2020,1000,50,100
1,2021,1100,52,110
2,2020,800,40,80
2,2021,900,42,90"""
        
        parsed = FileParser.parse_file_content(csv_content, "csv")
        
        assert parsed["data_type"] == "panel"
        
        converted = FileParser.convert_to_tool_format(parsed, "panel")
        
        assert len(converted["entity_ids"]) == 4
        assert len(converted["time_periods"]) == 4
        assert converted["y_variable"] == "profit"


class TestJSONExamples:
    """测试实际JSON示例"""
    
    def test_multivariate_data(self):
        """测试多变量数据"""
        json_content = {
            "GDP": [3.2, 2.8, 3.5, 2.9],
            "Inflation": [2.1, 2.3, 1.9, 2.4],
            "Unemployment": [4.5, 4.2, 4.0, 4.3]
        }
        
        parsed = FileParser.parse_file_content(
            json.dumps(json_content), "json"
        )
        
        assert parsed["n_variables"] == 3
        assert parsed["n_observations"] == 4
        assert parsed["data_type"] == "multivariate"


class TestErrorHandling:
    """测试错误处理"""
    
    def test_empty_csv(self):
        """测试空CSV"""
        with pytest.raises(ValueError, match="CSV文件为空"):
            FileParser.parse_file_content("", "csv")
    
    def test_invalid_json(self):
        """测试无效JSON"""
        with pytest.raises(ValueError, match="JSON格式错误"):
            FileParser.parse_file_content("{invalid json", "json")
    
    def test_unsupported_format(self):
        """测试不支持的格式"""
        with pytest.raises(ValueError, match="不支持的文件格式"):
            FileParser.parse_file_content("data", "xml")
    
    def test_no_numeric_data_csv(self):
        """测试无数值数据的CSV"""
        csv_content = """name,description
John,Engineer
Jane,Doctor"""
        
        with pytest.raises(ValueError, match="没有有效的数值数据"):
            FileParser.parse_file_content(csv_content, "csv")
    
    def test_regression_insufficient_variables(self):
        """测试回归分析变量不足"""
        parsed = {
            "data": {"x1": [1.0, 2.0]},
            "variables": ["x1"]
        }
        
        with pytest.raises(ValueError, match="至少需要2个变量"):
            FileParser.convert_to_tool_format(parsed, "regression")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])