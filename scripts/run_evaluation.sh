#!/bin/bash
set -e

echo "========================================"
echo "  一键模型对比评估"
echo "========================================"
echo ""

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo "⚠️  未找到 .env 文件，正在从模板创建..."
    cp .env.example .env
    echo "✅ 已创建 .env 文件，请编辑并填入 DASHSCOPE_API_KEY"
    echo ""
    read -p "是否已配置好 API Key? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ 请先配置 .env 文件中的 DASHSCOPE_API_KEY"
        exit 1
    fi
fi

echo "📦 安装依赖..."
uv pip install rouge-score bert-score openai tqdm python-dotenv

echo ""
echo "🚀 开始评估..."
echo ""

# 运行评估
python scripts/compare_models.py "$@"

echo ""
echo "✅ 评估完成！结果保存在 evaluation_results/ 目录"