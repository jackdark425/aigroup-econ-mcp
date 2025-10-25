# aigroup-econ-mcp v0.1.1 发布说明

## 🎉 发布信息

- **版本**: 0.1.1
- **发布日期**: 2025-10-25
- **PyPI链接**: https://pypi.org/project/aigroup-econ-mcp/

## 📦 已发布文件

✅ aigroup_econ_mcp-0.1.0-py3-none-any.whl (16.1 KB)
✅ aigroup_econ_mcp-0.1.0.tar.gz (26.9 KB)
✅ aigroup_econ_mcp-0.1.1-py3-none-any.whl (16.3 KB)
✅ aigroup_econ_mcp-0.1.1.tar.gz (28.5 KB)

## 🐛 重要修复

### 修复MCP连接失败的严重Bug

**问题**: 运行 `uvx aigroup-econ-mcp` 时出现 "MCP error -32000: Connection closed"

**根本原因**: CLI入口点结构混乱
- 同时使用 `@click.command()` 和 `@click.group()`
- 默认行为显示帮助而非启动MCP服务器
- MCP客户端期望stdio通信，但收到帮助文本导致连接关闭

**解决方案**:
- ✅ 完全重构CLI入口点为单一命令结构
- ✅ 确保默认行为正确启动stdio模式
- ✅ 修复Windows UTF-8编码问题
- ✅ 保留所有功能（版本、调试、多传输协议）

## ✅ 功能验证

所有7项功能测试通过：

| 测试项 | 状态 | 说明 |
|-------|------|------|
| 初始化服务器 | ✓ | MCP协议握手成功 |
| 工具列表 | ✓ | 5个计量经济学工具 |
| 资源列表 | ✓ | 资源系统正常 |
| 提示词列表 | ✓ | 1个分析模板 |
| 描述性统计 | ✓ | 计算成功 |
| 相关性分析 | ✓ | 分析成功 |
| 获取资源 | ✓ | 数据获取成功 |

## 📊 测试结果示例

### 描述性统计输出
```
均值: 0.0060
标准差: 0.0131
相关系数矩阵：
                stock_returns  market_returns
stock_returns          1.0000          0.9951
market_returns         0.9951          1.0000
```

### 相关性分析输出
```
Pearson相关系数矩阵：
              GDP_Growth  Inflation  Unemployment
GDP_Growth        1.0000    -0.9492       -0.3322
Inflation        -0.9492     1.0000        0.3514
Unemployment     -0.3322     0.3514        1.0000
```

## 🚀 安装使用

### 从PyPI安装（推荐）

```bash
# 使用uvx（推荐）
uvx aigroup-econ-mcp

# 使用pip
pip install aigroup-econ-mcp
```

### 与Claude Desktop集成

在配置文件中添加：

```json
{
  "mcpServers": {
    "aigroup-econ-mcp": {
      "command": "uvx",
      "args": ["aigroup-econ-mcp"]
    }
  }
}
```

## 📝 可用工具

1. **descriptive_statistics** - 描述性统计分析
2. **ols_regression** - OLS回归分析
3. **hypothesis_testing** - 假设检验
4. **time_series_analysis** - 时间序列分析
5. **correlation_analysis** - 相关性分析

## 🔗 相关链接

- **PyPI**: https://pypi.org/project/aigroup-econ-mcp/
- **GitHub**: https://github.com/aigroup/aigroup-econ-mcp
- **文档**: [README.md](README.md)
- **更新日志**: [CHANGELOG.md](CHANGELOG.md)

## 📧 联系方式

- **Email**: jackdark425@gmail.com
- **Issues**: https://github.com/aigroup/aigroup-econ-mcp/issues

## 🙏 致谢

感谢所有用户的支持和反馈！

---

**注意**: PyPI索引可能需要几分钟更新。如果遇到安装问题，请稍后重试或使用 `--refresh` 标志。