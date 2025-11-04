"""
计量经济学分析指导提示模板
提供分析步骤和最佳实践的指导提示
"""

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base


# 定义分析指导提示模板
def regression_analysis_guide(data_description: str) -> list[base.Message]:
    """生成回归分析指导提示"""
    return [
        base.SystemMessage(
            "You are an expert econometrics assistant helping with regression analysis."
        ),
        base.UserMessage(
            f"Help me conduct a proper regression analysis for the following data:\n\n{data_description}"
        ),
        base.AssistantMessage(
            """I'll guide you through the regression analysis process:

1. First, let's identify the appropriate model specification based on your data.
2. Then, we'll check the key assumptions of the classical linear regression model.
3. Next, we'll discuss potential diagnostic tests.
4. Finally, we'll interpret the results properly.

Let's start with understanding your data and research question."""
        )
    ]


def model_selection_prompt(data_type: str, research_question: str) -> list[base.Message]:
    """根据数据类型和研究问题推荐合适的计量经济学方法"""
    return [
        base.SystemMessage(
            "You are an expert econometrics methodologist."
        ),
        base.UserMessage(
            f"I have {data_type} data and want to answer: {research_question}\n\n"
            "What econometric methods would you recommend and why?"
        )
    ]