import json
import os
from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class JudgeScore:
    relevance: float
    factuality: float
    coverage: float
    overall: float
    reasoning: str


@dataclass
class LLMJudgeResult:
    scores: List[JudgeScore]
    avg_relevance: float
    avg_factuality: float
    avg_coverage: float
    avg_overall: float
    valid_count: int
    total_count: int


JUDGE_PROMPT_TEMPLATE = """You are an expert academic paper reviewer evaluator. Your task is to evaluate the quality of a paper review based on three criteria.

**Paper Content (excerpt):**
{paper_content}

**Ground Truth Review:**
{ground_truth}

**Model Generated Review:**
{generated_review}

Please evaluate the generated review on the following dimensions (1-10 scale):

1. **Relevance** (1-10): Does the review specifically address this paper, or is it generic/vague? A relevant review mentions specific methods, experiments, or claims from the paper.

2. **Factuality/Faithfulness** (1-10): Does the review accurately reflect the paper's content? Check for:
   - Misinterpretations of methods or experiments
   - Fabricated claims or conclusions
   - Incorrect understanding of the paper's contributions

3. **Coverage** (1-10): Does the review cover the key aspects of the paper?
   - Main contributions and novelty
   - Methodology
   - Experimental results
   - Limitations and weaknesses

Provide your evaluation in the following JSON format:
{{
    "relevance": <1-10>,
    "factuality": <1-10>,
    "coverage": <1-10>,
    "overall": <1-10>,
    "reasoning": "<Brief explanation of your scores>"
}}

Only output the JSON, no additional text."""


class LLMJudgeEvaluator:
    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.base_url = base_url or os.environ.get("DASHSCOPE_BASE_URL")
        self._client = None

    def _init_client(self):
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("openai not installed. Install with: pip install openai")
            except Exception as e:
                raise ValueError(
                    f"Failed to initialize OpenAI client: {e}. "
                    "Please set DASHSCOPE_API_KEY environment variable or pass api_key."
                )

    def truncate_text(self, text: str, max_chars: int = 4000) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "...[truncated]"

    def create_judge_prompt(
        self,
        paper_content: str,
        ground_truth: str,
        generated_review: str,
    ) -> str:
        return JUDGE_PROMPT_TEMPLATE.format(
            paper_content=self.truncate_text(paper_content, 3000),
            ground_truth=self.truncate_text(ground_truth, 2000),
            generated_review=self.truncate_text(generated_review, 2000),
        )

    def parse_judge_response(self, response: str) -> Optional[JudgeScore]:
        try:
            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return JudgeScore(
                    relevance=float(data.get("relevance", 0)),
                    factuality=float(data.get("factuality", 0)),
                    coverage=float(data.get("coverage", 0)),
                    overall=float(data.get("overall", 0)),
                    reasoning=data.get("reasoning", ""),
                )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        scores = {}
        for metric in ["relevance", "factuality", "coverage", "overall"]:
            pattern = rf'"{metric}"\s*:\s*(\d+(?:\.\d+)?)'
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                scores[metric] = float(match.group(1))

        if len(scores) >= 3:
            return JudgeScore(
                relevance=scores.get("relevance", 0),
                factuality=scores.get("factuality", 0),
                coverage=scores.get("coverage", 0),
                overall=scores.get("overall", 0),
                reasoning="",
            )
        return None

    def evaluate_single(
        self,
        paper_content: str,
        ground_truth: str,
        generated_review: str,
    ) -> Optional[JudgeScore]:
        self._init_client()

        prompt = self.create_judge_prompt(paper_content, ground_truth, generated_review)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            result_text = response.choices[0].message.content
            return self.parse_judge_response(result_text)
        except Exception as e:
            print(f"Error calling LLM judge: {e}")
            return None

    def evaluate_batch(
        self,
        paper_contents: List[str],
        ground_truths: List[str],
        generated_reviews: List[str],
        verbose: bool = True,
    ) -> LLMJudgeResult:
        scores = []

        for i, (paper, gt, gen) in enumerate(zip(paper_contents, ground_truths, generated_reviews)):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Judge evaluating sample {i + 1}/{len(paper_contents)}")

            score = self.evaluate_single(paper, gt, gen)
            if score:
                scores.append(score)

        if not scores:
            return LLMJudgeResult(
                scores=[],
                avg_relevance=0.0,
                avg_factuality=0.0,
                avg_coverage=0.0,
                avg_overall=0.0,
                valid_count=0,
                total_count=len(paper_contents),
            )

        return LLMJudgeResult(
            scores=scores,
            avg_relevance=sum(s.relevance for s in scores) / len(scores),
            avg_factuality=sum(s.factuality for s in scores) / len(scores),
            avg_coverage=sum(s.coverage for s in scores) / len(scores),
            avg_overall=sum(s.overall for s in scores) / len(scores),
            valid_count=len(scores),
            total_count=len(paper_contents),
        )
