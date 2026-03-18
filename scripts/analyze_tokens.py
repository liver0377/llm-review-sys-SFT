import json
import tiktoken
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TokenStats:
    count: int
    min_val: int
    max_val: int
    mean: float
    median: float
    p25: float
    p75: float
    p95: float
    p99: float


def get_tokenizer(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        print(f"无法加载 {model_name} tokenizer，使用 cl100k_base")
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, tokenizer) -> int:
    if hasattr(tokenizer, "encode"):
        result = tokenizer.encode(text, add_special_tokens=False)
        return len(result) if isinstance(result, list) else len(result.ids)
    return len(tokenizer.encode(text))


def analyze_data(json_path: str, tokenizer) -> Dict[str, List[int]]:
    print(f"\n加载数据: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instruction_tokens = []
    input_tokens = []
    output_tokens = []
    total_tokens = []

    for i, item in enumerate(data):
        if i % 1000 == 0:
            print(f"  处理进度: {i}/{len(data)}")

        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        inst_len = count_tokens(instruction, tokenizer)
        input_len = count_tokens(input_text, tokenizer)
        output_len = count_tokens(output, tokenizer)

        instruction_tokens.append(inst_len)
        input_tokens.append(input_len)
        output_tokens.append(output_len)
        total_tokens.append(inst_len + input_len + output_len)

    return {
        "instruction": instruction_tokens,
        "input": input_tokens,
        "output": output_tokens,
        "total": total_tokens,
    }


def compute_stats(values: List[int]) -> TokenStats:
    arr = np.array(values)
    return TokenStats(
        count=len(values),
        min_val=int(arr.min()),
        max_val=int(arr.max()),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        p25=float(np.percentile(arr, 25)),
        p75=float(np.percentile(arr, 75)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
    )


def print_stats(name: str, stats: TokenStats):
    print(f"\n{name}:")
    print(f"  样本数: {stats.count}")
    print(f"  最小值: {stats.min_val}")
    print(f"  最大值: {stats.max_val}")
    print(f"  平均值: {stats.mean:.1f}")
    print(f"  中位数: {stats.median:.1f}")
    print(f"  25%分位: {stats.p25:.1f}")
    print(f"  75%分位: {stats.p75:.1f}")
    print(f"  95%分位: {stats.p95:.1f}")
    print(f"  99%分位: {stats.p99:.1f}")


def main():
    data_dir = Path("data")
    tokenizer = get_tokenizer()

    print("\n" + "=" * 60)
    print("Token分布统计")
    print("=" * 60)

    files = {
        "训练集": "sft_train.json",
        "验证集": "sft_val.json",
        "测试集": "sft_test.json",
    }

    all_results = {}

    for name, filename in files.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"\n⚠️  文件不存在: {filepath}")
            continue

        results = analyze_data(str(filepath), tokenizer)
        all_results[name] = results

        print(f"\n{'=' * 60}")
        print(f"{name} ({filename})")
        print(f"{'=' * 60}")

        for key in ["instruction", "input", "output", "total"]:
            stats = compute_stats(results[key])
            label = {
                "instruction": "Instruction Tokens",
                "input": "Input Tokens",
                "output": "Output Tokens",
                "total": "Total Tokens",
            }[key]
            print_stats(label, stats)

    print("\n" + "=" * 60)
    print("总结建议")
    print("=" * 60)

    if "训练集" in all_results:
        train_stats = compute_stats(all_results["训练集"]["total"])
        print(f"\n推荐 cutoff_len: {int(train_stats.p99)} (覆盖99%样本)")
        print(f"或 cutoff_len: {int(train_stats.p95)} (覆盖95%样本)")

        input_stats = compute_stats(all_results["训练集"]["input"])
        output_stats = compute_stats(all_results["训练集"]["output"])
        print(f"\n输入平均长度: {input_stats.mean:.0f} tokens")
        print(f"输出平均长度: {output_stats.mean:.0f} tokens")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
