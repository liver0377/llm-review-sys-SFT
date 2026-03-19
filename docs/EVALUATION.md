# 模型对比评估框架

对SFT模型和Base模型进行全面对比评估，从三个维度评估模型性能。

## 评估维度

### 1. 格式遵循 (Format Compliance)
- 检查输出是否包含必需的四个部分：
  - Key Points
  - Strengths and Weaknesses
  - Suggestions for Improvement
  - Rating
- 检查是否正确输出Overall Quality和Review Confidence分数
- 输出格式合规率

### 2. 评审分数 (Score Evaluation)
- 提取模型输出的Overall Quality分数
- 与Ground Truth分数对比
- 回归指标：
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² (R-squared)

### 3. 评审质量 (Review Quality)

#### 自动化指标
- **ROUGE-L**: 衡量生成文本与参考文本的重叠度
- **BERTScore**: 基于BERT的语义相似度评估
  - Precision
  - Recall
  - F1

#### LLM-as-Judge
使用外部LLM对评审质量进行多维评分：

- **Relevance (相关性)**: 评审是否针对这篇论文，而非泛泛而谈
- **Factuality (正确性)**: 是否误读论文内容，是否捏造信息
- **Coverage (覆盖度)**: 是否覆盖论文的关键方面
- **Overall (综合评分)**: 综合评价

## 安装依赖

```bash
uv pip install rouge-score bert-score openai tqdm
```

## 使用方法

### 基本用法

```bash
python scripts/compare_models.py
```

### 自定义参数

```bash
python scripts/compare_models.py \
  --base_model_path pretrained/qwen2.5-7b-instruct \
  --sft_model_path pretrained/qwen2.5-7b-instruct-adapter \
  --test_data data/sft_test.json \
  --output_dir evaluation_results \
  --max_samples 100 \
  --llm_judge_model qwen-plus
```

### 禁用LLM-as-Judge

```bash
python scripts/compare_models.py --no_llm_judge
```

### 配置API

方式1: 使用.env文件（推荐）
```bash
# 在项目根目录创建 .env 文件
DASHSCOPE_API_KEY=your-api-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

python scripts/compare_models.py
```

方式2: 设置环境变量
```bash
export DASHSCOPE_API_KEY="your-api-key"
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 可选
python scripts/compare_models.py
```

方式3: 命令行参数
```bash
python scripts/compare_models.py \
  --api_key "your-api-key" \
  --base_url "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### 限制评估样本数量

```bash
# 仅评估前50个样本
python scripts/compare_models.py --max_samples 50

# LLM judge仅评估前20个样本
python scripts/compare_models.py --llm_judge_samples 20
```

## 输出结果

### 控制台输出

```
================================================================================
MODEL COMPARISON REPORT
================================================================================

Metric                              Base Model          SFT Model           Delta
--------------------------------------------------------------------------------
Format Compliance                       0.8500              0.9500         +0.1000
Score MSE (lower is better)             2.3400              1.1200         -1.2200
Score MAE (lower is better)             1.2300              0.8900         -0.3400
Score R² (higher is better)             0.4500              0.6700         +0.2200
ROUGE-L                                 0.3200              0.4100         +0.0900
BERTScore F1                            0.7500              0.8200         +0.0700
LLM Relevance                           6.5000              8.2000         +1.7000
LLM Factuality                          7.2000              8.5000         +1.3000
LLM Coverage                            6.8000              8.0000         +1.2000
LLM Overall                             6.8000              8.2000         +1.4000
================================================================================
```

### 文件输出

评估结果保存在 `evaluation_results/` 目录：

1. `base_predictions.json` - Base模型的预测结果
2. `sft_predictions.json` - SFT模型的预测结果
3. `comparison_YYYYMMDD_HHMMSS.json` - 详细对比结果

## 编程方式使用

```python
from src.evaluation.config import EvaluationConfig
from src.evaluation.comparator import ModelComparator

config = EvaluationConfig(
    test_data_path="data/sft_test.json",
    output_dir="evaluation_results",
    max_samples=100,
    llm_judge_model="qwen-plus",
    llm_judge_samples=20,
)

comparator = ModelComparator(config)
base_result, sft_result = comparator.run_comparison()
```

## 各模块单独使用

### 格式遵循评估

```python
from src.evaluation import FormatEvaluator

evaluator = FormatEvaluator()
result = evaluator.evaluate(response)
print(f"Format compliance: {result.format_compliance_score}")
```

### 分数评估

```python
from src.evaluation import ScoreEvaluator

evaluator = ScoreEvaluator()
result = evaluator.evaluate(predictions, ground_truths)
print(f"MSE: {result.mse}, MAE: {result.mae}, R²: {result.r2}")
```

### 质量评估

```python
from src.evaluation import QualityEvaluator

evaluator = QualityEvaluator()
result = evaluator.evaluate(predictions, references)
print(f"ROUGE-L: {result.rouge_l}, BERTScore F1: {result.bertscore_f1}")
```

### LLM-as-Judge

```python
from src.evaluation import LLMJudgeEvaluator

judge = LLMJudgeEvaluator(
    model="qwen-plus",
    api_key="your-api-key"
)
result = judge.evaluate_single(paper, ground_truth, prediction)
print(f"Relevance: {result.relevance}, Factuality: {result.factuality}")
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base_model_path` | `pretrained/qwen2.5-7b-instruct` | Base模型路径 |
| `--sft_model_path` | `pretrained/qwen2.5-7b-instruct-adapter` | SFT模型路径(LoRA) |
| `--test_data` | `data/sft_test.json` | 测试数据路径 |
| `--output_dir` | `evaluation_results` | 结果输出目录 |
| `--max_samples` | `None` | 最大评估样本数 |
| `--llm_judge_model` | `qwen-plus` | Judge使用的LLM |
| `--llm_judge_samples` | `None` | Judge评估样本数 |
| `--no_llm_judge` | `False` | 禁用LLM-as-Judge |
| `--api_key` | `None` | Dashscope API Key |
| `--base_url` | `None` | Dashscope API Base URL |