import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

from llamafactory.chat import ChatModel
from tqdm import tqdm

from src.evaluation.config import EvaluationConfig, EvaluationResult, ModelConfig
from src.evaluation.format_eval import FormatEvaluator
from src.evaluation.llm_judge import LLMJudgeEvaluator
from src.evaluation.quality_eval import QualityEvaluator
from src.evaluation.score_eval import ScoreEvaluator


class ModelComparator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.format_evaluator = FormatEvaluator()
        self.score_evaluator = ScoreEvaluator()
        self.quality_evaluator = QualityEvaluator()
        self.llm_judge = None

        os.makedirs(config.output_dir, exist_ok=True)

    def load_test_data(self) -> List[Dict]:
        with open(self.config.test_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if self.config.max_samples:
            data = data[: self.config.max_samples]
        return data

    def create_chat_model(self, model_config: ModelConfig) -> ChatModel:
        args = {
            "model_name_or_path": model_config.model_name_or_path,
            "template": model_config.template,
        }

        if model_config.model_type == "lora" and model_config.adapter_name_or_path:
            args["adapter_name_or_path"] = model_config.adapter_name_or_path
            args["finetuning_type"] = model_config.finetuning_type or "lora"

        return ChatModel(args)

    def generate_predictions(
        self,
        model_config: ModelConfig,
        test_data: List[Dict],
    ) -> List[str]:
        print(f"\nLoading model: {model_config.name}")
        chat_model = self.create_chat_model(model_config)

        predictions = []
        print(f"Generating predictions for {len(test_data)} samples...")

        for item in tqdm(test_data, desc=f"Generating ({model_config.name})"):
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")

            full_prompt = f"{instruction}\n\n{input_text}"
            messages = [{"role": "user", "content": full_prompt}]

            response = ""
            for new_text in chat_model.stream_chat(messages):
                response += new_text

            predictions.append(response)

        return predictions

    def evaluate_model(
        self,
        model_config: ModelConfig,
        test_data: List[Dict],
        predictions: List[str],
        ground_truths: List[str],
        paper_contents: List[str],
    ) -> EvaluationResult:
        print(f"\nEvaluating model: {model_config.name}")

        result = EvaluationResult(model_name=model_config.name)

        if self.config.enable_format_eval:
            print("  Evaluating format compliance...")
            avg_compliance, format_results = self.format_evaluator.evaluate_batch(predictions)
            result.format_compliance = avg_compliance

            coverage = self.format_evaluator.get_section_coverage(format_results)
            print(f"    Format compliance: {avg_compliance:.4f}")
            for section, cov in coverage.items():
                print(f"    {section}: {cov:.4f}")

        if self.config.enable_score_eval:
            print("  Evaluating score predictions...")
            score_result = self.score_evaluator.evaluate(predictions, ground_truths)
            result.score_mse = score_result.mse
            result.score_mae = score_result.mae
            result.score_r2 = score_result.r2
            print(f"    MSE: {score_result.mse:.4f}")
            print(f"    MAE: {score_result.mae:.4f}")
            print(f"    R²: {score_result.r2:.4f}")
            print(f"    Valid samples: {score_result.valid_count}/{score_result.total_count}")

        if self.config.enable_quality_eval:
            print("  Evaluating review quality (BERTScore & ROUGE-L)...")
            quality_result = self.quality_evaluator.evaluate(predictions, ground_truths)
            result.rouge_l = quality_result.rouge_l
            result.bertscore_precision = quality_result.bertscore_precision
            result.bertscore_recall = quality_result.bertscore_recall
            result.bertscore_f1 = quality_result.bertscore_f1
            print(f"    ROUGE-L: {quality_result.rouge_l:.4f}")
            print(f"    BERTScore F1: {quality_result.bertscore_f1:.4f}")

        if self.config.enable_llm_judge and self.config.llm_judge_enabled:
            print("  Evaluating with LLM-as-judge...")

            judge_samples = self.config.llm_judge_samples or len(predictions)
            judge_samples = min(judge_samples, len(predictions))

            if self.llm_judge is None:
                self.llm_judge = LLMJudgeEvaluator(
                    model=self.config.llm_judge_model,
                    api_key=self.config.llm_judge_api_key,
                    base_url=self.config.llm_judge_base_url,
                )

            judge_result = self.llm_judge.evaluate_batch(
                paper_contents[:judge_samples],
                ground_truths[:judge_samples],
                predictions[:judge_samples],
                verbose=True,
            )

            result.llm_relevance = judge_result.avg_relevance
            result.llm_factuality = judge_result.avg_factuality
            result.llm_coverage = judge_result.avg_coverage
            result.llm_overall = judge_result.avg_overall

            print(f"    Relevance: {judge_result.avg_relevance:.2f}/10")
            print(f"    Factuality: {judge_result.avg_factuality:.2f}/10")
            print(f"    Coverage: {judge_result.avg_coverage:.2f}/10")
            print(f"    Overall: {judge_result.avg_overall:.2f}/10")
            print(f"    Valid samples: {judge_result.valid_count}/{judge_result.total_count}")

        return result

    def save_predictions(
        self,
        model_name: str,
        predictions: List[str],
        test_data: List[Dict],
    ):
        output_path = os.path.join(self.config.output_dir, f"{model_name}_predictions.json")
        output_data = []
        for item, pred in zip(test_data, predictions):
            output_data.append(
                {
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "ground_truth": item.get("output", ""),
                    "prediction": pred,
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Predictions saved to: {output_path}")

    def save_results(
        self,
        base_result: EvaluationResult,
        sft_result: EvaluationResult,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_path = os.path.join(self.config.output_dir, f"comparison_{timestamp}.json")
        results_data = {
            "config": {
                "test_data_path": self.config.test_data_path,
                "max_samples": self.config.max_samples,
                "llm_judge_model": self.config.llm_judge_model,
                "evaluation_time": timestamp,
            },
            "base_model": asdict(base_result),
            "sft_model": asdict(sft_result),
        }

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {results_path}")

    def print_comparison_report(
        self,
        base_result: EvaluationResult,
        sft_result: EvaluationResult,
    ):
        print("\n" + "=" * 80)
        print("MODEL COMPARISON REPORT")
        print("=" * 80)

        print(f"\n{'Metric':<30} {'Base Model':>20} {'SFT Model':>20} {'Delta':>15}")
        print("-" * 80)

        metrics = [
            ("Format Compliance", "format_compliance"),
            ("Score MSE (lower is better)", "score_mse"),
            ("Score MAE (lower is better)", "score_mae"),
            ("Score R² (higher is better)", "score_r2"),
            ("ROUGE-L", "rouge_l"),
            ("BERTScore F1", "bertscore_f1"),
            ("LLM Relevance", "llm_relevance"),
            ("LLM Factuality", "llm_factuality"),
            ("LLM Coverage", "llm_coverage"),
            ("LLM Overall", "llm_overall"),
        ]

        for display_name, attr_name in metrics:
            base_val = getattr(base_result, attr_name, None)
            sft_val = getattr(sft_result, attr_name, None)

            if base_val is not None and sft_val is not None:
                delta = sft_val - base_val
                delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
                print(f"{display_name:<30} {base_val:>20.4f} {sft_val:>20.4f} {delta_str:>15}")
            elif base_val is not None:
                print(f"{display_name:<30} {base_val:>20.4f} {'N/A':>20} {'N/A':>15}")
            elif sft_val is not None:
                print(f"{display_name:<30} {'N/A':>20} {sft_val:>20.4f} {'N/A':>15}")

        print("=" * 80)

        print("\nSummary:")
        improvements = []
        regressions = []

        if base_result.format_compliance and sft_result.format_compliance:
            if sft_result.format_compliance > base_result.format_compliance:
                improvements.append("Format Compliance")
            elif sft_result.format_compliance < base_result.format_compliance:
                regressions.append("Format Compliance")

        if base_result.score_mse and sft_result.score_mse:
            if sft_result.score_mse < base_result.score_mse:
                improvements.append("Score Prediction (MSE)")
            elif sft_result.score_mse > base_result.score_mse:
                regressions.append("Score Prediction (MSE)")

        if base_result.bertscore_f1 and sft_result.bertscore_f1:
            if sft_result.bertscore_f1 > base_result.bertscore_f1:
                improvements.append("Review Quality (BERTScore)")
            elif sft_result.bertscore_f1 < base_result.bertscore_f1:
                regressions.append("Review Quality (BERTScore)")

        if base_result.llm_overall and sft_result.llm_overall:
            if sft_result.llm_overall > base_result.llm_overall:
                improvements.append("LLM Judge Overall Score")
            elif sft_result.llm_overall < base_result.llm_overall:
                regressions.append("LLM Judge Overall Score")

        if improvements:
            print(f"  ✓ SFT model improved on: {', '.join(improvements)}")
        if regressions:
            print(f"  ✗ SFT model regressed on: {', '.join(regressions)}")
        print()

    def run_comparison(self):
        print("=" * 80)
        print("MODEL COMPARISON EVALUATION")
        print("=" * 80)

        test_data = self.load_test_data()
        print(f"\nLoaded {len(test_data)} test samples")

        ground_truths = [item.get("output", "") for item in test_data]
        paper_contents = [item.get("input", "") for item in test_data]

        base_predictions = self.generate_predictions(self.config.base_model, test_data)

        sft_predictions = self.generate_predictions(self.config.sft_model, test_data)

        if self.config.save_predictions:
            self.save_predictions("base", base_predictions, test_data)
            self.save_predictions("sft", sft_predictions, test_data)

        base_result = self.evaluate_model(
            self.config.base_model,
            test_data,
            base_predictions,
            ground_truths,
            paper_contents,
        )

        sft_result = self.evaluate_model(
            self.config.sft_model,
            test_data,
            sft_predictions,
            ground_truths,
            paper_contents,
        )

        self.print_comparison_report(base_result, sft_result)

        if self.config.save_detailed_results:
            self.save_results(base_result, sft_result)

        return base_result, sft_result
