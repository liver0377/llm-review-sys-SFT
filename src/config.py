import os
from dataclasses import dataclass, field
from typing import Optional

os.environ["FORCE_TORCHRUN"] = "1"

@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    template: str = "qwen"


@dataclass
class LoRAConfig:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target: str = "all"


@dataclass
class TrainingConfig:
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_scheduler: str = "cosine"

    cutoff_len: int = 12569
    logging_steps: int = 10
    save_steps: int = 500

    use_gradient_checkpointing: bool = True
    optimizer: str = "adamw_8bit"
    use_deepspeed: bool = True


@dataclass
class QuantizationConfig:
    enabled: bool = False
    bits: int = 4
    quant_type: str = "nf4"


@dataclass
class DatasetConfig:
    name: str = "paper_review"
    dir: str = "data"
    eval_dataset: Optional[str] = None


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    output_dir: str = "outputs/paper_review_qwen2.5_7b_lora"
    seed: int = 42

    precision: str = "bf16"

    max_steps: Optional[int] = None

    def to_dict(self) -> dict:
        config = {
            "stage": "sft",
            "do_train": True,
            "model_name_or_path": self.model.name,
            "dataset": self.dataset.name,
            "dataset_dir": self.dataset.dir,
            "template": self.model.template,
            "finetuning_type": "lora",
            "lora_target": self.lora.target,
            "lora_rank": self.lora.rank,
            "lora_alpha": self.lora.alpha,
            "lora_dropout": self.lora.dropout,
            "output_dir": self.output_dir,
            "overwrite_output_dir": True,
            "cutoff_len": self.training.cutoff_len,
            "preprocessing_num_workers": 4,
            "per_device_train_batch_size": self.training.batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "lr_scheduler_type": self.training.lr_scheduler,
            "logging_steps": self.training.logging_steps,
            "save_steps": self.training.save_steps,
            "learning_rate": self.training.learning_rate,
            "num_train_epochs": self.training.num_epochs,
            "max_grad_norm": self.training.max_grad_norm,
            "gradient_checkpointing": self.training.use_gradient_checkpointing,
            "optim": self.training.optimizer,
            "seed": self.seed,
        }

        if self.precision == "bf16":
            config["bf16"] = True
        elif self.precision == "fp16":
            config["fp16"] = True

        if self.quantization.enabled:
            config["quantization_bit"] = self.quantization.bits
            config["quantization_type"] = self.quantization.quant_type

        if self.max_steps is not None:
            config["max_steps"] = self.max_steps

        if self.training.use_deepspeed:
            config["deepspeed"] = "configs/deepspeed_zero2.json"

        if self.dataset.eval_dataset:
            config["eval_dataset"] = self.dataset.eval_dataset
            config["per_device_eval_batch_size"] = 1
            config["eval_strategy"] = "steps"
            config["eval_steps"] = self.training.save_steps
            config["load_best_model_at_end"] = True
            config["metric_for_best_model"] = "eval_loss"
            config["greater_is_better"] = False

        return config


def get_local_test_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.precision = "fp16"
    config.quantization.enabled = True
    config.training.batch_size = 1
    config.training.gradient_accumulation_steps = 1
    config.training.cutoff_len = 4096
    config.max_steps = 3
    config.output_dir = "outputs/test_run"
    return config


def get_server_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.precision = "bf16"
    config.dataset.eval_dataset = "paper_review_val"
    return config


def get_qlora_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.precision = "bf16"
    config.quantization.enabled = True
    config.dataset.eval_dataset = "paper_review_val"
    config.output_dir = "outputs/paper_review_qwen2.5_7b_qlora"
    return config
