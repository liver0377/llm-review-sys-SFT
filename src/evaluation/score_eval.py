import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class ScoreEvaluationResult:
    mse: float
    mae: float
    r2: float
    predictions: List[Optional[float]]
    ground_truths: List[Optional[float]]
    valid_count: int
    total_count: int


class ScoreEvaluator:
    def __init__(self):
        pass

    def extract_score(self, response: str) -> Optional[float]:
        patterns = [
            r"Overall\s*Quality:?\s*(\d+(?:\.\d+)?)",
            r"Overall Quality[^:]*:\s*(\d+(?:\.\d+)?)",
            r"Rating[^:]*:\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    if 1 <= score <= 10:
                        return score
                except ValueError:
                    continue
        return None

    def extract_gt_score(self, output: str) -> Optional[float]:
        return self.extract_score(output)

    def evaluate(self, predictions: List[str], ground_truths: List[str]) -> ScoreEvaluationResult:
        pred_scores = []
        gt_scores = []

        for pred, gt in zip(predictions, ground_truths):
            pred_score = self.extract_score(pred)
            gt_score = self.extract_gt_score(gt)

            if pred_score is not None and gt_score is not None:
                pred_scores.append(pred_score)
                gt_scores.append(gt_score)

        if len(pred_scores) == 0:
            return ScoreEvaluationResult(
                mse=float("inf"),
                mae=float("inf"),
                r2=float("-inf"),
                predictions=[],
                ground_truths=[],
                valid_count=0,
                total_count=len(predictions),
            )

        pred_arr = np.array(pred_scores)
        gt_arr = np.array(gt_scores)

        mse = float(np.mean((pred_arr - gt_arr) ** 2))
        mae = float(np.mean(np.abs(pred_arr - gt_arr)))

        ss_res = np.sum((gt_arr - pred_arr) ** 2)
        ss_tot = np.sum((gt_arr - np.mean(gt_arr)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return ScoreEvaluationResult(
            mse=mse,
            mae=mae,
            r2=float(r2),
            predictions=pred_scores,
            ground_truths=gt_scores,
            valid_count=len(pred_scores),
            total_count=len(predictions),
        )

    def get_score_distribution(self, predictions: List[str]) -> dict:
        scores = []
        for pred in predictions:
            score = self.extract_score(pred)
            if score is not None:
                scores.append(score)

        if not scores:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "median": None,
            }

        scores_arr = np.array(scores)
        return {
            "mean": float(np.mean(scores_arr)),
            "std": float(np.std(scores_arr)),
            "min": float(np.min(scores_arr)),
            "max": float(np.max(scores_arr)),
            "median": float(np.median(scores_arr)),
        }
