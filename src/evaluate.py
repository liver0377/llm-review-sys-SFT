import json
import os
from typing import Dict, List, Optional


def load_trainer_state(output_dir: str) -> Optional[Dict]:
    state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        return None

    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(state: Dict) -> Dict:
    metrics = {
        "total_steps": state.get("global_step", 0),
        "train_loss": [],
        "eval_loss": [],
    }

    for entry in state.get("log_history", []):
        step = entry.get("step", 0)

        if "loss" in entry:
            metrics["train_loss"].append({"step": step, "loss": entry["loss"]})

        if "eval_loss" in entry:
            metrics["eval_loss"].append({"step": step, "loss": entry["eval_loss"]})

    return metrics


def print_evaluation_report(output_dir: str = "outputs/paper_review_qwen2.5_7b_lora"):
    print("\n" + "=" * 60)
    print("训练评估报告")
    print("=" * 60)

    state = load_trainer_state(output_dir)

    if not state:
        print(f"\n❌ 未找到训练状态文件: {output_dir}/trainer_state.json")
        print("请先运行训练")
        return

    metrics = extract_metrics(state)

    print(f"\n📊 训练统计")
    print(f"  总步数: {metrics['total_steps']}")

    if metrics["train_loss"]:
        latest_train_loss = metrics["train_loss"][-1]["loss"]
        print(f"  最终训练Loss: {latest_train_loss:.4f}")

    if metrics["eval_loss"]:
        print(f"\n📈 验证集Loss变化:")
        for entry in metrics["eval_loss"]:
            print(f"  Step {entry['step']:4d}: {entry['loss']:.4f}")

        best_eval = min(metrics["eval_loss"], key=lambda x: x["loss"])
        print(f"\n  最佳验证Loss: {best_eval['loss']:.4f} (Step {best_eval['step']})")

    print("\n" + "=" * 60)
    print("💡 后续步骤:")
    print("  1. 运行 chat.py 进行交互式测试")
    print("  2. 在测试集上评估生成质量")
    print("  3. 计算ROUGE/BLEU等指标")
    print("=" * 60 + "\n")


def check_model_exists(output_dir: str) -> bool:
    adapter_path = os.path.join(output_dir, "adapter_config.json")
    return os.path.exists(adapter_path)


if __name__ == "__main__":
    import sys

    output_dir = (
        sys.argv[1] if len(sys.argv) > 1 else "outputs/paper_review_qwen2.5_7b_lora"
    )
    print_evaluation_report(output_dir)
