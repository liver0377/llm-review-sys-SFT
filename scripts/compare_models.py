#!/usr/bin/env python
import argparse
import os

from dotenv import load_dotenv

from src.evaluation.config import EvaluationConfig
from src.evaluation.comparator import ModelComparator


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Compare SFT model with Base model")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="pretrained/qwen2.5-7b-instruct",
        help="Path to base model",
    )
    parser.add_argument(
        "--sft_model_path",
        type=str,
        default="pretrained/qwen2.5-7b-instruct-adapter",
        help="Path to SFT model (LoRA adapter)",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/sft_test.json",
        help="Path to test data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None for all)",
    )
    parser.add_argument(
        "--llm_judge_model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for judge evaluation",
    )
    parser.add_argument(
        "--llm_judge_samples",
        type=int,
        default=None,
        help="Number of samples for LLM judge evaluation (None for all)",
    )
    parser.add_argument(
        "--no_llm_judge",
        action="store_true",
        help="Disable LLM-as-judge evaluation",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Dashscope API key (or set DASHSCOPE_API_KEY env var)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Dashscope API base URL (or set DASHSCOPE_BASE_URL env var)",
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    base_url = args.base_url or os.environ.get("DASHSCOPE_BASE_URL")

    config = EvaluationConfig(
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        llm_judge_model=args.llm_judge_model,
        llm_judge_samples=args.llm_judge_samples,
        llm_judge_enabled=not args.no_llm_judge,
        llm_judge_api_key=api_key,
        llm_judge_base_url=base_url,
    )

    config.base_model.model_name_or_path = args.base_model_path
    config.sft_model.adapter_name_or_path = args.sft_model_path

    comparator = ModelComparator(config)
    comparator.run_comparison()


if __name__ == "__main__":
    main()
