import json
import tiktoken
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib


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


def visualize_results(all_results: Dict, output_dir: str = "output"):
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Token Distribution Analysis", fontsize=16)

    token_types = ["instruction", "input", "output", "total"]
    labels = ["Instruction Tokens", "Input Tokens", "Output Tokens", "Total Tokens"]
    colors = {"训练集": "#3498db", "验证集": "#2ecc71", "测试集": "#e74c3c"}

    for idx, (token_type, label) in enumerate(zip(token_types, labels)):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Token Count")

        dataset_names = []
        means = []
        medians = []
        p25s = []
        p75s = []
        mins = []
        maxs = []

        for name in ["训练集", "验证集", "测试集"]:
            if name in all_results and token_type in all_results[name]:
                stats = compute_stats(all_results[name][token_type])
                dataset_names.append(name)
                means.append(stats.mean)
                medians.append(stats.median)
                p25s.append(stats.p25)
                p75s.append(stats.p75)
                mins.append(stats.min_val)
                maxs.append(stats.max_val)

        if dataset_names:
            x = np.arange(len(dataset_names))
            width = 0.6

            bars = ax.bar(x, means, width, label="Mean", color=[colors[n] for n in dataset_names])

            for i, (bar, p25, p75) in enumerate(zip(bars, p25s, p75s)):
                ax.errorbar(
                    bar.get_x() + bar.get_width() / 2,
                    means[i],
                    yerr=[[means[i] - p25s[i]], [p75s[i] - means[i]]],
                    fmt="none",
                    color="black",
                    capsize=5,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(dataset_names)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "token_stats_summary.png", dpi=150, bbox_inches="tight")
    print(f"\n📊 统计图已保存: {output_path / 'token_stats_summary.png'}")
    plt.close()

    for name in all_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{name} - Token Distribution", fontsize=16)

        for idx, (token_type, label) in enumerate(zip(token_types, labels)):
            ax = axes[idx // 2, idx % 2]
            ax.set_title(label, fontsize=12)

            data = all_results[name][token_type]
            ax.hist(data, bins=50, color="#3498db", edgecolor="white", alpha=0.7)

            stats = compute_stats(data)
            ax.axvline(
                stats.mean,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {stats.mean:.0f}",
            )
            ax.axvline(
                stats.median,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Median: {stats.median:.0f}",
            )
            ax.axvline(
                stats.p95,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"P95: {stats.p95:.0f}",
            )
            ax.axvline(
                stats.p99,
                color="purple",
                linestyle="--",
                linewidth=2,
                label=f"P99: {stats.p99:.0f}",
            )

            ax.set_xlabel("Token Count")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        safe_name = name.replace("/", "_")
        plt.savefig(output_path / f"{safe_name}_distribution.png", dpi=150, bbox_inches="tight")
        print(f"📊 {name} 分布图已保存: {output_path / f'{safe_name}_distribution.png'}")
        plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Token Distribution Comparison (Box Plot)", fontsize=16)

    for idx, (token_type, label) in enumerate(zip(token_types, labels)):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(label, fontsize=12)

        box_data = []
        box_labels = []
        box_colors = []

        for name in ["训练集", "验证集", "测试集"]:
            if name in all_results and token_type in all_results[name]:
                box_data.append(all_results[name][token_type])
                box_labels.append(name)
                box_colors.append(colors[name])

        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_ylabel("Token Count")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "token_boxplot.png", dpi=150, bbox_inches="tight")
    print(f"📊 箱线图已保存: {output_path / 'token_boxplot.png'}")
    plt.close()


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

    if all_results:
        visualize_results(all_results)


if __name__ == "__main__":
    main()
