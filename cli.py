"""
AIGroup 计量经济学 MCP 服务器命令行接口
"""

import sys
import click
import uvicorn
from . import mcp


@click.group()
def cli():
    """AIGroup 计量经济学 MCP 服务器
    
    支持 stdio、HTTP 和 SSE 传输协议的 MCP 服务器
    """
    pass


@cli.command()
@click.option('--port', default=8000, help='HTTP 服务器端口')
@click.option('--host', default='127.0.0.1', help='HTTP 服务器主机')
@click.option('--transport', default='stdio',
              type=click.Choice(['stdio', 'streamable-http', 'sse']),
              help='传输协议 (默认: stdio)')
@click.option('--debug', is_flag=True, help='调试模式')
@click.option('--mount-path', default=None, help='挂载路径')
def serve(port: int, host: str, transport: str, debug: bool, mount_path: str):
    """启动 MCP 服务器"""
    
    # 获取 MCP 服务器实例
    mcp_server = mcp

    # 调试模式
    if debug:
        click.echo(f"[DEBUG] 调试模式已启用", err=True)

    # 根据传输协议启动服务器
    if transport == 'stdio':
        # stdio 传输 - 直接使用标准输入输出进行 MCP 通信
        # 错误信息输出到 stderr
        from . import __version__
        click.echo(f"[INFO] aigroup-econ-mcp v{__version__} 启动中...", err=True)
        click.echo(f"[INFO] 传输协议: stdio (MCP 协议)", err=True)
        if debug:
            click.echo(f"[DEBUG] 调试模式已启用", err=True)
        click.echo(f"[INFO] 服务器准备就绪。等待 MCP 客户端连接...", err=True)
        mcp_server.run(transport='stdio')
        
    elif transport == 'streamable-http':
        # Streamable HTTP - 使用 uvicorn 启动
        click.echo(f"[INFO] 启动 aigroup-econ-mcp 服务器", err=True)
        click.echo(f"[INFO] 专业计量经济学 MCP 工具，用于 AI 数据分析", err=True)
        click.echo(f"[INFO] 传输协议: {transport}", err=True)
        click.echo(f"[INFO] 服务地址: http://{host}:{port}", err=True)
        if mount_path:
            click.echo(f"[INFO] 挂载路径: {mount_path}", err=True)
        
        # Starlette 应用 - 使用 uvicorn 运行
        app = mcp_server.streamable_http_app()
        uvicorn.run(app, host=host, port=port, log_level="info")
        
    elif transport == 'sse':
        # SSE - 使用 uvicorn 启动
        click.echo(f"[INFO] 启动 aigroup-econ-mcp 服务器", err=True)
        click.echo(f"[INFO] 专业计量经济学 MCP 工具，用于 AI 数据分析", err=True)
        click.echo(f"[INFO] 传输协议: {transport}", err=True)
        click.echo(f"[INFO] 服务地址: http://{host}:{port}", err=True)
        if mount_path:
            click.echo(f"[INFO] 挂载路径: {mount_path}", err=True)
        
        # Starlette 应用 - 使用 uvicorn 运行
        app = mcp_server.sse_app()
        uvicorn.run(app, host=host, port=port, log_level="info")


@cli.command()
def version():
    """显示版本信息"""
    from . import __version__
    click.echo(f"aigroup-econ-mcp v{__version__}", err=True)
    click.echo("专业计量经济学 MCP 工具", err=True)
    click.echo("作者: AIGroup", err=True)


if __name__ == "__main__":
    cli()