# PyPI 发布指南

## 发布前提条件

### 1. 配置 PyPI 认证
在用户主目录创建 `.pypirc` 文件：

**文件位置**: `C:\Users\用户名\.pypirc`
**内容**:
```ini
[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcCJ*
```

### 2. 确保系统Python已安装twine
```bash
pip install twine
```

## 完整发布流程

### 步骤1: 更新版本号
需要更新两个文件中的版本号：

#### 1.1 更新 `pyproject.toml`
```toml
[project]
name = "aigroup_econ_mcp"
version = "X.X.X"  # 更新版本号
```

#### 1.2 更新 `__init__.py`
```python
__version__ = "X.X.X"  # 更新版本号
```

### 步骤2: 构建包
```bash
uv build
```

### 步骤3: 发布到PyPI
使用系统Python执行发布命令：
```bash
C:\Python313\python.exe -m twine upload dist/aigroup_econ_mcp-版本号*
```

**具体示例**:
```bash
# 对于版本 2.0.2
C:\Python313\python.exe -m twine upload dist/aigroup_econ_mcp-2.0.2*
```

## 验证发布结果

发布成功后，可以在以下链接查看：
- **PyPI项目页面**: https://pypi.org/project/aigroup-econ-mcp/
- **具体版本页面**: https://pypi.org/project/aigroup-econ-mcp/版本号/

## 安装验证

用户可以通过以下命令安装最新版本：
```bash
pip install aigroup-econ-mcp==版本号
```

## 注意事项

1. **版本号格式**: 遵循语义化版本控制 (Semantic Versioning)
2. **认证文件**: 确保 `.pypirc` 文件在用户主目录且包含正确的token
3. **Python路径**: 使用系统Python (`C:\Python313\python.exe`) 而不是虚拟环境Python
4. **包文件**: 发布前确保 `dist/` 目录中有正确版本的包文件
5. **版本一致性**: 确保 `pyproject.toml` 和 `__init__.py` 中的版本号一致

## 故障排除

### 如果发布失败：
1. 检查 `.pypirc` 文件中的token是否正确
2. 确认版本号在PyPI上不存在（避免重复发布）
3. 检查网络连接
4. 验证包文件是否成功构建
5. 确认 `pyproject.toml` 和 `__init__.py` 中的版本号一致

### 如果twine命令未找到：
```bash
# 安装twine
pip install twine
# 或使用uv安装
uv add twine
```

## 自动化脚本（可选）

可以创建发布脚本 `publish.bat`：
```batch
@echo off
echo 开始发布流程...
echo 1. 构建包...
uv build
echo 2. 发布到PyPI...
C:\Python313\python.exe -m twine upload dist/aigroup_econ_mcp-%1*
echo 发布完成！
pause
```

使用方法：
```bash
publish.bat 2.0.2
```

---

**重要提醒**: 每次发布前请确保：
- ✅ `pyproject.toml` 中的版本号已更新
- ✅ `__init__.py` 中的版本号已更新
- ✅ 代码已测试
- ✅ 文档已更新（如果需要）
- ✅ 认证配置正确