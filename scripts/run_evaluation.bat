@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   一键模型对比评估
echo ========================================
echo.

REM 检查 .env 文件
if not exist ".env" (
    echo ⚠️  未找到 .env 文件，正在从模板创建...
    copy .env.example .env >nul
    echo ✅ 已创建 .env 文件
    echo.
    echo 请编辑 .env 文件并填入 DASHSCOPE_API_KEY
    echo.
    pause
    exit /b 1
)

echo 📦 安装依赖...
uv pip install rouge-score bert-score openai tqdm python-dotenv

echo.
echo 🚀 开始评估...
echo.

REM 运行评估
python scripts/compare_models.py %*

echo.
echo ✅ 评估完成！结果保存在 evaluation_results/ 目录
pause