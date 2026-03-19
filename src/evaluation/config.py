from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    name: str
    model_type: str
    model_name_or_path: str
    adapter_name_or_path: Optional[str] = None
    template: str = "qwen"
    finetuning_type: Optional[str] = None


@dataclass
class EvaluationConfig:
    base_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="base",
            model_type="base",
            model_name_or_path="pretrained/qwen2.5-7b-instruct",
            template="qwen",
        )
    )
    sft_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="sft",
            model_type="lora",
            model_name_or_path="pretrained/qwen2.5-7b-instruct",
            adapter_name_or_path="pretrained/qwen2.5-7b-instruct-adapter",
            template="qwen",
            finetuning_type="lora",
        )
    )
    test_data_path: str = "data/sft_test.json"
    output_dir: str = "evaluation_results"
    max_samples: Optional[int] = None
    batch_size: int = 1
    max_length: int = 4096

    llm_judge_enabled: bool = True
    llm_judge_model: str = "qwen-plus"
    llm_judge_api_key: Optional[str] = None
    llm_judge_base_url: Optional[str] = None
    llm_judge_samples: Optional[int] = None

    enable_format_eval: bool = True
    enable_score_eval: bool = True
    enable_quality_eval: bool = True
    enable_llm_judge: bool = True

    save_predictions: bool = True
    save_detailed_results: bool = True


@dataclass
class EvaluationResult:
    model_name: str
    format_compliance: Optional[float] = None
    score_mse: Optional[float] = None
    score_mae: Optional[float] = None
    score_r2: Optional[float] = None
    rouge_l: Optional[float] = None
    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    bertscore_f1: Optional[float] = None
    llm_relevance: Optional[float] = None
    llm_factuality: Optional[float] = None
    llm_coverage: Optional[float] = None
    llm_overall: Optional[float] = None
