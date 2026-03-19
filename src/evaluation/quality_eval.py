from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class QualityMetrics:
    rouge_l: float
    bertscore_precision: float
    bertscore_recall: float
    bertscore_f1: float


@dataclass
class QualityEvaluationResult:
    rouge_l: float
    bertscore_precision: float
    bertscore_recall: float
    bertscore_f1: float
    sample_count: int


class QualityEvaluator:
    def __init__(self, bertscore_model: str = "microsoft/deberta-xlarge-mnli"):
        self.bertscore_model = bertscore_model
        self._rouge = None
        self._bertscore = None

    def _init_rouge(self):
        if self._rouge is None:
            try:
                from rouge_score import rouge_scorer

                self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            except ImportError:
                raise ImportError(
                    "rouge_score not installed. Install with: pip install rouge-score"
                )

    def _init_bertscore(self):
        if self._bertscore is None:
            try:
                from bert_score import BERTScorer

                self._bertscore = BERTScorer(
                    model_type=self.bertscore_model, lang="en", rescale_with_baseline=True
                )
            except ImportError:
                raise ImportError("bert_score not installed. Install with: pip install bert-score")

    def compute_rouge_l(self, prediction: str, reference: str) -> float:
        self._init_rouge()
        scores = self._rouge.score(reference, prediction)
        return scores["rougeL"].fmeasure

    def compute_bertscore(self, predictions: List[str], references: List[str]) -> tuple:
        self._init_bertscore()
        p, r, f1 = self._bertscore.score(predictions, references)
        return (float(p.mean().item()), float(r.mean().item()), float(f1.mean().item()))

    def extract_review_content(self, text: str) -> str:
        sections_to_extract = [
            "Key Points",
            "Strengths and Weaknesses",
            "Suggestions for Improvement",
        ]

        extracted = []
        lines = text.split("\n")
        current_section = None
        section_content = []

        for line in lines:
            is_section_header = False
            for section in sections_to_extract:
                if re.search(rf"###\s*{re.escape(section)}", line, re.IGNORECASE):
                    if current_section and section_content:
                        extracted.append("\n".join(section_content))
                    current_section = section
                    section_content = [line]
                    is_section_header = True
                    break

            if not is_section_header and current_section:
                if re.search(r"###\s*Rating", line, re.IGNORECASE):
                    if section_content:
                        extracted.append("\n".join(section_content))
                    current_section = None
                    section_content = []
                else:
                    section_content.append(line)

        if current_section and section_content:
            extracted.append("\n".join(section_content))

        if extracted:
            return "\n\n".join(extracted)
        return text

    def evaluate(
        self, predictions: List[str], references: List[str], extract_content: bool = True
    ) -> QualityEvaluationResult:
        if extract_content:
            processed_preds = [self.extract_review_content(p) for p in predictions]
            processed_refs = [self.extract_review_content(r) for r in references]
        else:
            processed_preds = predictions
            processed_refs = references

        rouge_scores = []
        for pred, ref in zip(processed_preds, processed_refs):
            score = self.compute_rouge_l(pred, ref)
            rouge_scores.append(score)
        avg_rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

        bs_precision, bs_recall, bs_f1 = self.compute_bertscore(processed_preds, processed_refs)

        return QualityEvaluationResult(
            rouge_l=avg_rouge_l,
            bertscore_precision=bs_precision,
            bertscore_recall=bs_recall,
            bertscore_f1=bs_f1,
            sample_count=len(predictions),
        )

    def evaluate_sample(
        self, prediction: str, reference: str, extract_content: bool = True
    ) -> QualityMetrics:
        if extract_content:
            pred = self.extract_review_content(prediction)
            ref = self.extract_review_content(reference)
        else:
            pred = prediction
            ref = reference

        rouge_l = self.compute_rouge_l(pred, ref)

        p, r, f1 = self.compute_bertscore([pred], [ref])

        return QualityMetrics(
            rouge_l=rouge_l, bertscore_precision=p, bertscore_recall=r, bertscore_f1=f1
        )
