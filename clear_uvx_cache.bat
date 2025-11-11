@echo off
REM 清除 uvx 缓存 - Windows 批处理脚本
REM 解决 uvx 总是使用旧版本的问题

echo ============================================================
echo 🧹 uvx 缓存清除工具 (Windows)
echo    解决 aigroup-econ-mcp 总是运行旧版本的问题
echo ============================================================
echo.

set CACHE_PATH=%LOCALAPPDATA%\uv\cache\wheels

echo 🔍 缓存路径: %CACHE_PATH%
echo.

if exist "%CACHE_PATH%" (
    echo 🗑️  正在删除缓存目录...
    rmdir /s /q "%CACHE_PATH%"
    if %ERRORLEVEL% EQU 0 (
        echo ✅ 缓存清除成功！
        echo.
        echo 💡 现在可以运行: uvx aigroup-econ-mcp
    ) else (
        echo ❌ 清除缓存失败，请使用管理员权限运行
    )
) else (
    echo ✅ 缓存目录不存在，无需清除
)

echo.
echo ============================================================
echo 📝 后续步骤:
echo    1. 运行: uvx aigroup-econ-mcp
echo    2. 或者: uvx --no-cache aigroup-econ-mcp
echo    3. 验证版本: uvx aigroup-econ-mcp --version
echo ============================================================
echo.

pause