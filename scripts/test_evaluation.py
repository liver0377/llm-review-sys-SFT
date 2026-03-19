#!/usr/bin/env python
import json

from src.evaluation import FormatEvaluator, ScoreEvaluator


def test_format_evaluator():
    print("Testing FormatEvaluator...")
    evaluator = FormatEvaluator()

    test_response = """### Key Points
This paper presents a novel approach to...

### Strengths and Weaknesses
**Strengths:**
- Novel methodology
- Strong experiments

**Weaknesses:**
- Limited evaluation

### Suggestions for Improvement
Add more baselines.

### Rating
**Overall Quality:** 7.0
**Review Confidence:** 4.0"""

    result = evaluator.evaluate(test_response)
    print(f"  Format compliance: {result.format_compliance_score:.2f}")
    print(f"  Has Key Points: {result.has_key_points}")
    print(f"  Has Strengths/Weaknesses: {result.has_strengths_weaknesses}")
    print(f"  Has Suggestions: {result.has_suggestions}")
    print(f"  Has Rating: {result.has_rating}")
    print(f"  Has Overall Quality: {result.has_overall_quality}")
    print(f"  Has Confidence: {result.has_confidence}")
    assert result.format_compliance_score > 0.8, "Format compliance should be high"
    print("  ✓ FormatEvaluator test passed\n")


def test_score_evaluator():
    print("Testing ScoreEvaluator...")
    evaluator = ScoreEvaluator()

    predictions = [
        "Overall Quality: 7.0",
        "Overall Quality: 5.5",
        "The paper is good. Overall Quality: 8.0",
    ]
    ground_truths = [
        "Overall Quality: 7.5",
        "Overall Quality: 6.0",
        "Overall Quality: 7.0",
    ]

    result = evaluator.evaluate(predictions, ground_truths)
    print(f"  MSE: {result.mse:.4f}")
    print(f"  MAE: {result.mae:.4f}")
    print(f"  R²: {result.r2:.4f}")
    print(f"  Valid samples: {result.valid_count}/{result.total_count}")
    assert result.valid_count == 3, "Should extract 3 valid scores"
    print("  ✓ ScoreEvaluator test passed\n")


def test_data_loading():
    print("Testing data loading...")
    with open("data/sft_test.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} test samples")

    sample = data[0]
    assert "instruction" in sample, "Should have instruction"
    assert "input" in sample, "Should have input"
    assert "output" in sample, "Should have output"
    print("  ✓ Data loading test passed\n")


def main():
    print("=" * 60)
    print("Evaluation Framework Quick Test")
    print("=" * 60 + "\n")

    test_data_loading()
    test_format_evaluator()
    test_score_evaluator()

    print("=" * 60)
    print("All tests passed! Framework is ready to use.")
    print("=" * 60)


if __name__ == "__main__":
    main()
