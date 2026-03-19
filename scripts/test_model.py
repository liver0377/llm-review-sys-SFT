#!/usr/bin/env python3
"""
测试微调后的模型在测试集上的效果 (使用vLLM加速)
"""

import json
import argparse
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def load_test_data(data_path: str, num_samples: int = 2):
    """加载测试数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:num_samples]


def format_input(sample: dict) -> str:
    """格式化输入"""
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def apply_chat_template(prompt: str) -> str:
    """应用Qwen对话模板"""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def test_model(
    model_path: str,
    test_data_path: str = "data/sft_test.json",
    num_samples: int = 2,
    output_file: str = None,
):
    """测试模型"""
    base_model = "Qwen/Qwen2.5-7B-Instruct"

    print(f"\n加载模型: {base_model}")
    print(f"LoRA路径: {model_path}")

    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_lora_rank=32,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )

    print(f"加载测试数据: {test_data_path}")
    test_data = load_test_data(test_data_path, num_samples)
    print(f"测试样本数: {len(test_data)}\n")

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
    )

    lora_request = LoRARequest("paper_review", 1, model_path)

    prompts = []
    ground_truths = []
    for sample in test_data:
        user_input = format_input(sample)
        prompt = apply_chat_template(user_input)
        prompts.append(prompt)
        ground_truths.append(sample.get("output", ""))

    print("生成中...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    results = []
    for i, (output, ground_truth) in enumerate(zip(outputs, ground_truths)):
        print("=" * 80)
        print(f"样本 {i + 1}/{len(test_data)}")
        print("=" * 80)

        response = output.outputs[0].text.strip()

        print("\n[模型输出]")
        print("-" * 80)
        print(response)
        print()

        print("\n[期望输出]")
        print("-" * 80)
        print(ground_truth[:500] + "..." if len(ground_truth) > 500 else ground_truth)
        print()

        results.append(
            {
                "sample_id": i + 1,
                "model_output": response,
                "ground_truth": ground_truth,
            }
        )

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="测试微调后的模型 (vLLM)")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/paper_review_qwen2.5_7b_qlora/checkpoint-500",
        help="LoRA模型路径",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/sft_test.json",
        help="测试数据路径",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="测试样本数量",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（可选）",
    )

    args = parser.parse_args()

    test_model(
        model_path=args.model_path,
        test_data_path=args.test_data,
        num_samples=args.num_samples,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
