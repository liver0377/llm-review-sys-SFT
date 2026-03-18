# 论文评审模型 SFT 训练

基于 Qwen2.5-7B-Instruct 的论文评审模型微调项目

## 项目结构

```
llm-review-sys-SFT/
├── data/                    # 数据集
│   ├── dataset_info.json    # 数据集配置
│   ├── sft_train.json       # 训练集 (9018)
│   ├── sft_val.json         # 验证集 (1061)
│   └── sft_test.json        # 测试集 (531)
├── src/                     # 核心代码
│   ├── config.py           # 配置管理
│   ├── train.py            # 训练逻辑
│   ├── chat.py             # 对话推理
│   └── evaluate.py         # 评估工具
├── scripts/                 # 快捷脚本
│   ├── test_local.py       # 本地测试
│   ├── train_qlora.py      # QLoRA训练
│   ├── chat.py             # 对话测试
│   └── evaluate.py         # 评估报告
├── configs/                 # 配置文件 (可选)
├── outputs/                 # 模型输出
├── main.py                  # 主入口
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
uv venv

# 激活环境
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 安装依赖
uv pip install llamafactory transformers datasets accelerate peft trl bitsandbytes scipy einops sentencepiece tiktoken
```

### 2. 本地测试 (Windows 4060)

```bash
python scripts/test_local.py
```

**配置特点:**
- ✅ fp16 精度
- ✅ 4bit 量化
- ✅ 仅运行3步快速验证
- ✅ 显存需求: 6GB+

### 3. 服务器训练

```bash
# 标准训练 (16GB+ 显存)
python main.py

# QLoRA训练 (8GB+ 显存)
python scripts/train_qlora.py
```

## 使用方式

### 方式1: 快捷脚本 (推荐)

```bash
# 本地测试
python scripts/test_local.py

# 服务器训练
python main.py

# QLoRA训练
python scripts/train_qlora.py

# 对话测试
python scripts/chat.py

# 查看训练结果
python scripts/evaluate.py
```

### 方式2: 模块调用

```bash
# 本地测试
python -m src.train test

# 服务器训练
python -m src.train server

# QLoRA训练
python -m src.train qlora
```

### 方式3: 自定义配置

```python
from src.config import ExperimentConfig, train

config = ExperimentConfig()
config.training.learning_rate = 5e-5
config.training.num_epochs = 5
config.output_dir = "outputs/custom_run"

train(config)
```

## 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | Qwen2.5-7B-Instruct | 基座模型 |
| 方法 | LoRA | 参数高效微调 |
| LoRA Rank | 64 | 低秩矩阵维度 |
| Learning Rate | 1e-4 | 学习率 |
| Batch Size | 2 | 单卡batch size |
| Gradient Accumulation | 8 | 梯度累积 |
| Epochs | 3 | 训练轮数 |
| Cutoff Length | 8192 | 最大序列长度 |

## 配置说明

### 核心配置类

- `ExperimentConfig`: 总配置
- `ModelConfig`: 模型配置
- `LoRAConfig`: LoRA参数
- `TrainingConfig`: 训练超参
- `QuantizationConfig`: 量化配置
- `DatasetConfig`: 数据集配置

### 预设配置

```python
from src.config import get_local_test_config, get_server_config, get_qlora_config

# 本地测试
config = get_local_test_config()

# 服务器训练
config = get_server_config()

# QLoRA训练
config = get_qlora_config()
```

## 对话测试

```bash
python scripts/chat.py

# 指定模型路径
python scripts/chat.py outputs/paper_review_qwen2.5_7b_lora
```

## 评估结果

```bash
python scripts/evaluate.py

# 指定输出目录
python scripts/evaluate.py outputs/custom_run
```

## 注意事项

1. **首次运行**会自动下载模型 (约15GB)
2. **网络要求**: 能访问 HuggingFace
3. **显存需求**: 
   - 本地测试: 6GB+
   - QLoRA训练: 8GB+
   - 标准训练: 16GB+
4. **训练时间**: 约3-5小时 (取决于硬件)

## 常见问题

**Q: bf16 报错**
A: 使用 `scripts/test_local.py` (已改为 fp16)

**Q: CUDA OOM**
A: 降低 `batch_size` 或使用 QLoRA

**Q: 数据集加载失败**
A: 检查 `data/dataset_info.json` 路径

## License

MIT