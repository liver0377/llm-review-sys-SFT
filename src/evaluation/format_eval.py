import re
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class FormatCheckResult:
    has_key_points: bool
    has_strengths_weaknesses: bool
    has_suggestions: bool
    has_rating: bool
    has_overall_quality: bool
    has_confidence: bool
    format_compliance_score: float
    details: Dict[str, Any]


class FormatEvaluator:
    def __init__(self):
        self.required_sections = [
            "Key Points",
            "Strengths and Weaknesses",
            "Suggestions for Improvement",
            "Rating",
        ]

    def evaluate(self, response: str) -> FormatCheckResult:
        results = {
            "key_points": self._check_section(response, "Key Points"),
            "strengths_weaknesses": self._check_strengths_weaknesses(response),
            "suggestions": self._check_section(response, "Suggestions for Improvement"),
            "rating": self._check_rating_section(response),
        }

        overall_quality, confidence = self._extract_scores(response)
        results["overall_quality_extracted"] = overall_quality is not None
        results["confidence_extracted"] = confidence is not None

        total_checks = len(results)
        passed_checks = sum(1 for v in results.values() if v)
        compliance_score = passed_checks / total_checks if total_checks > 0 else 0.0

        return FormatCheckResult(
            has_key_points=results["key_points"],
            has_strengths_weaknesses=results["strengths_weaknesses"],
            has_suggestions=results["suggestions"],
            has_rating=results["rating"],
            has_overall_quality=overall_quality is not None,
            has_confidence=confidence is not None,
            format_compliance_score=compliance_score,
            details=results,
        )

    def _check_section(self, text: str, section_name: str) -> bool:
        patterns = [
            rf"###\s*{re.escape(section_name)}",
            rf"\*\*{re.escape(section_name)}\*\*",
            rf"{re.escape(section_name)}:",
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _check_strengths_weaknesses(self, text: str) -> bool:
        has_strengths = bool(re.search(r"\*\*Strengths?\*\*:?", text, re.IGNORECASE)) or bool(
            re.search(r"Strengths?:", text, re.IGNORECASE)
        )
        has_weaknesses = bool(re.search(r"\*\*Weaknesses?\*\*:?", text, re.IGNORECASE)) or bool(
            re.search(r"Weaknesses?:", text, re.IGNORECASE)
        )
        return has_strengths and has_weaknesses

    def _check_rating_section(self, text: str) -> bool:
        has_rating_header = self._check_section(text, "Rating")
        has_overall = bool(re.search(r"Overall\s*Quality:?\s*\d+", text, re.IGNORECASE))
        has_confidence = bool(re.search(r"Review\s*Confidence:?\s*\d+", text, re.IGNORECASE))
        return has_rating_header or (has_overall and has_confidence)

    def _extract_scores(self, text: str) -> tuple:
        overall_pattern = r"Overall\s*Quality:?\s*(\d+(?:\.\d+)?)"
        confidence_pattern = r"Review\s*Confidence:?\s*(\d+(?:\.\d+)?)"

        overall_match = re.search(overall_pattern, text, re.IGNORECASE)
        confidence_match = re.search(confidence_pattern, text, re.IGNORECASE)

        overall = float(overall_match.group(1)) if overall_match else None
        confidence = float(confidence_match.group(1)) if confidence_match else None

        return overall, confidence

    def evaluate_batch(self, responses: List[str]) -> tuple[float, List[FormatCheckResult]]:
        results = []
        for response in responses:
            result = self.evaluate(response)
            results.append(result)

        avg_compliance = (
            sum(r.format_compliance_score for r in results) / len(results) if results else 0.0
        )
        return avg_compliance, results

    def get_section_coverage(self, results: List[FormatCheckResult]) -> Dict[str, float]:
        if not results:
            return {}

        coverage = {
            "key_points": sum(1 for r in results if r.has_key_points) / len(results),
            "strengths_weaknesses": sum(1 for r in results if r.has_strengths_weaknesses)
            / len(results),
            "suggestions": sum(1 for r in results if r.has_suggestions) / len(results),
            "rating": sum(1 for r in results if r.has_rating) / len(results),
            "overall_quality": sum(1 for r in results if r.has_overall_quality) / len(results),
            "confidence": sum(1 for r in results if r.has_confidence) / len(results),
        }
        return coverage
