#!/bin/bash
# 清除 uvx 缓存 - Shell 脚本 (macOS/Linux)
# 解决 uvx 总是使用旧版本的问题

echo "============================================================"
echo "🧹 uvx 缓存清除工具 (macOS/Linux)"
echo "   解决 aigroup-econ-mcp 总是运行旧版本的问题"
echo "============================================================"
echo ""

CACHE_PATH="$HOME/.cache/uv/wheels"

echo "🔍 缓存路径: $CACHE_PATH"
echo ""

if [ -d "$CACHE_PATH" ]; then
    echo "🗑️  正在删除缓存目录..."
    rm -rf "$CACHE_PATH"
    if [ $? -eq 0 ]; then
        echo "✅ 缓存清除成功！"
        echo ""
        echo "💡 现在可以运行: uvx aigroup-econ-mcp"
    else
        echo "❌ 清除缓存失败，请检查权限"
    fi
else
    echo "✅ 缓存目录不存在，无需清除"
fi

echo ""
echo "============================================================"
echo "📝 后续步骤:"
echo "   1. 运行: uvx aigroup-econ-mcp"
echo "   2. 或者: uvx --no-cache aigroup-econ-mcp"
echo "   3. 验证版本: uvx aigroup-econ-mcp --version"
echo "============================================================"
echo ""