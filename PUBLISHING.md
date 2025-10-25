# PyPI 发布指南

## 已完成的准备工作

✅ 版本号已更新到 0.1.1
✅ 包已成功构建
✅ 创建了更新日志 (CHANGELOG.md)
✅ 所有功能测试通过

## 构建产物

```
dist/
├── aigroup_econ_mcp-0.1.1.tar.gz          # 源码分发包
└── aigroup_econ_mcp-0.1.1-py3-none-any.whl # 二进制wheel包
```

## 发布到PyPI

### 方法1: 使用uv发布（推荐）

```bash
# 发布到PyPI（需要PyPI账号和Token）
uv publish

# 或指定token
uv publish --token YOUR_PYPI_TOKEN
```

### 方法2: 使用twine发布

```bash
# 安装twine
pip install twine

# 检查包
twine check dist/*

# 上传到TestPyPI（测试）
twine upload --repository testpypi dist/*

# 上传到PyPI（正式）
twine upload dist/*
```

## 配置PyPI凭据

### 选项1: 使用环境变量

```bash
# Windows PowerShell
$env:UV_PUBLISH_TOKEN="pypi-..."

# Linux/Mac
export UV_PUBLISH_TOKEN="pypi-..."
```

### 选项2: 使用.pypirc文件

创建 `~/.pypirc` 文件：

```ini
[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE
```

## 获取PyPI Token

1. 访问 https://pypi.org/manage/account/token/
2. 登录你的PyPI账号
3. 创建新的API Token
4. 复制token（格式: pypi-...）

## 发布后验证

```bash
# 安装发布的包
pip install aigroup-econ-mcp

# 或使用uvx
uvx aigroup-econ-mcp --version
```

## 发布检查清单

- [x] 版本号已更新
- [x] 包已构建成功
- [x] 所有测试通过
- [x] README.md完整
- [x] LICENSE存在
- [x] CHANGELOG.md已创建
- [ ] PyPI凭据已配置
- [ ] 执行发布命令
- [ ] 验证安装

## 注意事项

⚠️ **重要**: 
- 发布到PyPI后不能删除或修改
- 建议先发布到TestPyPI测试
- 确保版本号遵循语义化版本规范
- 每次发布必须使用新的版本号

## 快速发布命令

```bash
# 一键发布（需要已配置token）
cd d:/aigroup-econ-mcp && uv publish