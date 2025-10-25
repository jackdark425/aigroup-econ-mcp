# 发布 aigroup-econ-mcp v0.1.3 到 PyPI

## 📋 发布前检查清单

- [x] 版本号已更新: `pyproject.toml` (0.1.2 → 0.1.3)
- [x] 更新日志已完成: `CHANGELOG.md`
- [x] 发布说明已创建: `RELEASE_NOTES_v0.1.3.md`
- [x] Bug修复报告已完成: `BUG_FIX_REPORT.md`
- [x] 代码修复已完成并测试
- [ ] 所有测试通过
- [ ] Git提交并推送

## 🚀 发布步骤

### 步骤1: 清理旧的构建文件
```bash
# 清理dist目录
rm -rf dist/
rm -rf build/
rm -rf *.egg-info
```

### 步骤2: 构建包
```bash
# 使用hatch构建（推荐）
python -m build

# 或使用传统方式
# python setup.py sdist bdist_wheel
```

验证构建结果：
```bash
ls -lh dist/
# 应该看到:
# aigroup-econ-mcp-0.1.3.tar.gz
# aigroup_econ_mcp-0.1.3-py3-none-any.whl
```

### 步骤3: 测试包（可选但推荐）
```bash
# 在虚拟环境中测试安装
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate
pip install dist/aigroup_econ_mcp-0.1.3-py3-none-any.whl

# 测试命令
aigroup-econ-mcp --version

# 测试导入
python -c "from aigroup_econ_mcp import server; print('Import OK')"

# 退出测试环境
deactivate
```

### 步骤4: 上传到 TestPyPI（可选）
```bash
# 首先安装twine
pip install twine

# 上传到TestPyPI测试
python -m twine upload --repository testpypi dist/*

# 测试从TestPyPI安装
pip install --index-url https://test.pypi.org/simple/ aigroup-econ-mcp==0.1.3
```

### 步骤5: 上传到 PyPI（正式发布）
```bash
# 上传到PyPI
python -m twine upload dist/*

# 输入PyPI凭据:
# Username: __token__
# Password: pypi-... (你的API token)
```

### 步骤6: 验证发布
```bash
# 等待几分钟后测试安装
pip install --upgrade aigroup-econ-mcp

# 验证版本
pip show aigroup-econ-mcp
# Version: 0.1.3

# 测试uvx安装（推荐方式）
uvx aigroup-econ-mcp
```

### 步骤7: Git标签和推送
```bash
# 提交所有更改
git add .
git commit -m "Release v0.1.3: Fix NumPy serialization and OLS data shape issues"

# 创建标签
git tag -a v0.1.3 -m "Version 0.1.3 - Critical bug fixes for OLS and time series tools"

# 推送到远程仓库
git push origin main
git push origin v0.1.3
```

### 步骤8: GitHub Release（可选）
1. 访问 GitHub 仓库的 Releases 页面
2. 点击 "Draft a new release"
3. 选择标签 v0.1.3
4. 标题: "v0.1.3 - Critical Bug Fixes"
5. 描述: 复制 `RELEASE_NOTES_v0.1.3.md` 的内容
6. 附加文件: 
   - `BUG_FIX_REPORT.md`
   - dist文件（可选）
7. 点击 "Publish release"

## 🔧 故障排除

### 问题1: twine上传失败
```bash
# 检查PyPI token是否正确
# 确保在 ~/.pypirc 中配置了token

# 或使用环境变量
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
python -m twine upload dist/*
```

### 问题2: 版本冲突
```bash
# 如果版本已存在，需要增加版本号
# 编辑 pyproject.toml，将版本改为 0.1.4
# 然后重新构建和上传
```

### 问题3: 包验证失败
```bash
# 检查包内容
python -m tarfile -l dist/aigroup-econ-mcp-0.1.3.tar.gz

# 检查wheel内容
python -m zipfile -l dist/aigroup_econ_mcp-0.1.3-py3-none-any.whl
```

## ✅ 发布后验证

### 1. PyPI页面检查
访问: https://pypi.org/project/aigroup-econ-mcp/0.1.3/
- 检查版本显示正确
- 检查描述和文档显示正确
- 检查依赖项列表

### 2. 安装测试
```bash
# 创建新环境测试
python -m venv fresh_install
source fresh_install/bin/activate

# 从PyPI安装
pip install aigroup-econ-mcp==0.1.3

# 运行测试
python test_simple_ols.py

# 清理
deactivate
rm -rf fresh_install
```

### 3. MCP客户端测试
```bash
# 使用uvx测试（推荐）
uvx aigroup-econ-mcp

# 在Claude Desktop或其他MCP客户端中测试
# 确保工具列表包含5个工具
# 测试ols_regression和time_series_analysis
```

## 📝 发布检查清单

完成后勾选：

- [ ] 包成功构建
- [ ] TestPyPI测试通过（可选）
- [ ] PyPI发布成功
- [ ] 版本号在PyPI上正确显示
- [ ] 从PyPI安装测试通过
- [ ] Git提交和标签已推送
- [ ] GitHub Release已创建（可选）
- [ ] MCP客户端测试通过
- [ ] 通知用户升级

## 📢 发布公告

发布成功后，可以在以下渠道发布公告：

1. **GitHub Discussions/Issues**
   - 关闭相关的bug报告issue
   - 在Discussions中发布更新公告

2. **文档更新**
   - 更新README.md中的安装说明
   - 更新示例代码（如果有变化）

3. **社区通知**
   - MCP社区论坛
   - 相关技术社区

## 🎉 发布完成！

恭喜！v0.1.3已成功发布。用户现在可以通过以下方式获取更新：

```bash
# PyPI
pip install --upgrade aigroup-econ-mcp

# uvx（推荐）
uvx aigroup-econ-mcp  # 自动使用最新版本
```

---

**下一个版本规划**: v0.1.4
- 考虑添加更多计量经济学工具
- 改进文档和示例
- 性能优化