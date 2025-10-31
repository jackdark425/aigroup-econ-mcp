"""
自动为所有工具添加file_path参数
"""

# 需要在这些工具的参数列表开头添加file_path参数
tools_to_fix = [
    # 面板数据工具
    "panel_fixed_effects",
    "panel_random_effects", 
    "panel_hausman_test",
    "panel_unit_root_test",
    # 高级时间序列工具
    "var_model_analysis",
    "vecm_model_analysis",
    "garch_model_analysis",
    "state_space_model_analysis",
    "variance_decomposition_analysis",
    # 机器学习工具
    "random_forest_regression_analysis",
    "gradient_boosting_regression_analysis",
    "lasso_regression_analysis",
    "ridge_regression_analysis",
    "cross_validation_analysis",
    "feature_importance_analysis_tool"
]

file_path_param = '''    file_path: Annotated[Optional[str], Field(default=None, description="文件路径")] = None,
'''

print(f"需要为{len(tools_to_fix)}个工具添加file_path参数")
print("请手动在每个工具的参数列表中添加file_path参数（在ctx参数之后）")