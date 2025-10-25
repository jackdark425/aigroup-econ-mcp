"""
AIGroup 计量经济学 MCP 服务命令行入口
"""

import sys
import click
from .server import create_mcp_server


@click.command()
@click.option('--port', default=8000, help='服务器端口')
@click.option('--host', default='127.0.0.1', help='服务器地址')
@click.option('--transport', default='stdio',
              type=click.Choice(['stdio', 'streamable-http', 'sse']),
              help='传输协议 (默认: stdio)')
@click.option('--debug', is_flag=True, help='启用调试模式')
@click.option('--mount-path', default=None, help='挂载路径')
@click.option('--version', is_flag=True, help='显示版本信息')
def cli(port: int, host: str, transport: str, debug: bool, mount_path: str, version: bool):
    """AIGroup 计量经济学 MCP 工具
    
    默认以stdio模式启动MCP服务器，适用于MCP客户端集成。
    """
    
    # 处理版本标志
    if version:
        click.echo("aigroup-econ-mcp v0.1.0", err=True)
        click.echo("Professional econometrics MCP tool", err=True)
        click.echo("Author: AIGroup", err=True)
        sys.exit(0)

    # 创建MCP服务器
    mcp_server = create_mcp_server()

    # 设置调试模式
    if debug:
        mcp_server.settings.debug = True
        click.echo(f"[DEBUG] 调试模式已启用", err=True)

    # 根据传输协议启动服务器
    if transport == 'stdio':
        # stdio模式直接运行，不输出任何日志到stdout（MCP协议通信）
        # 所有日志输出到stderr
        if debug:
            click.echo(f"[DEBUG] Starting in stdio mode", err=True)
        mcp_server.run(transport='stdio')
        
    elif transport == 'streamable-http':
        # Streamable HTTP模式
        click.echo(f"[INFO] Starting aigroup-econ-mcp server", err=True)
        click.echo(f"[INFO] Professional econometrics MCP tool for AI data analysis", err=True)
        click.echo(f"[INFO] Transport protocol: {transport}", err=True)
        click.echo(f"[INFO] Service address: http://{host}:{port}", err=True)
        if mount_path:
            click.echo(f"[INFO] Mount path: {mount_path}", err=True)
            
        mcp_server.run(
            transport='streamable-http',
            host=host,
            port=port,
            mount_path=mount_path or '/mcp'
        )
        
    elif transport == 'sse':
        # SSE模式
        click.echo(f"[INFO] Starting aigroup-econ-mcp server", err=True)
        click.echo(f"[INFO] Professional econometrics MCP tool for AI data analysis", err=True)
        click.echo(f"[INFO] Transport protocol: {transport}", err=True)
        click.echo(f"[INFO] Service address: http://{host}:{port}", err=True)
        if mount_path:
            click.echo(f"[INFO] Mount path: {mount_path}", err=True)
            
        mcp_server.run(
            transport='sse',
            host=host,
            port=port,
            mount_path=mount_path or '/sse'
        )


if __name__ == "__main__":
    cli()