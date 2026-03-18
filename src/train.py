from llamafactory.train.tuner import run_exp
from src.config import (
    ExperimentConfig,
    get_local_test_config,
    get_server_config,
    get_qlora_config,
)


def train(config: ExperimentConfig, description: str = ""):
    if description:
        print("\n" + "=" * 60)
        print(description)
        print("=" * 60 + "\n")

    args = config.to_dict()

    print(f"Model: {config.model.name}")
    print(f"Dataset: {config.dataset.name}")
    print(f"Precision: {config.precision}")
    print(
        f"Quantization: {'Enabled ({}bit)'.format(config.quantization.bits) if config.quantization.enabled else 'Disabled'}"
    )
    print(f"Output: {config.output_dir}")
    print()

    run_exp(args)


def train_local_test():
    config = get_local_test_config()
    train(
        config,
        "本地测试模式\n- 仅运行3步验证代码\n- 使用4bit量化 + fp16\n- cutoff_len=4096",
    )


def train_server():
    config = get_server_config()
    train(config, "服务器训练模式\n- 完整训练配置\n- bf16精度\n- 验证集监控")


def train_qlora():
    config = get_qlora_config()
    train(config, "服务器训练模式 (QLoRA)\n- 4bit量化\n- 适合显存较小的服务器")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  python -m src.train test      # 本地测试")
        print("  python -m src.train server    # 服务器训练")
        print("  python -m src.train qlora     # QLoRA训练")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "test":
        train_local_test()
    elif mode == "server":
        train_server()
    elif mode == "qlora":
        train_qlora()
    else:
        print(f"未知模式: {mode}")
        print("支持的模式: test, server, qlora")
        sys.exit(1)
