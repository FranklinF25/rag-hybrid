"""DeepEval harness: RAG metrics over the golden set.

Run:
    DEEPSEEK_API_KEY=sk-... pytest -m eval evals/

Prerequisites (the suite skips with a reason otherwise):
- local Qdrant up and the sample collection indexed
  (``rag-hybrid index sample/ --recreate``)
- ``DEEPSEEK_API_KEY`` in the environment (or ``.env``)

Thresholds follow the research baseline: faithfulness >= 0.85 and
contextual precision >= 0.75 are pinned; contextual recall and answer
relevancy use 0.70 as v1 starting points.

All models are stateless API clients (Qwen3 embeddings and rerank-3-lite
via OpenRouter, DeepSeek for generation): the single module-scoped query
pipeline holds no local weights; each golden question reuses it. The
retrieval context is taken from the same run (the reranked documents), so
no second pipeline is built.
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from openai import NOT_GIVEN, OpenAI

from rag_hybrid.config import get_settings
from rag_hybrid.retrieval import ask, build_query_pipeline

GOLDEN_SET = Path(__file__).parent / "golden_set.csv"

pytestmark = pytest.mark.eval


def _prerequisite_reason() -> str | None:
    settings = get_settings()
    if not settings.deepseek_api_key:
        return "DEEPSEEK_API_KEY is not set"
    try:
        url = f"{settings.qdrant_url.rstrip('/')}/collections/{settings.qdrant_collection}"
        with urllib.request.urlopen(url, timeout=3) as response:
            points = json.loads(response.read())["result"]["points_count"]
        if points == 0:
            return f"collection '{settings.qdrant_collection}' is empty: run `rag-hybrid index sample/ --recreate`"
    except OSError:
        return f"Qdrant not reachable at {settings.qdrant_url} (docker compose up -d)"
    except (KeyError, ValueError):
        return f"unexpected response from Qdrant for collection '{settings.qdrant_collection}'"
    return None


_PREREQUISITE_REASON = _prerequisite_reason()
if _PREREQUISITE_REASON:
    pytest.skip(_PREREQUISITE_REASON, allow_module_level=True)


class DeepSeekJudge(DeepEvalBaseLLM):
    """DeepSeek as the DeepEval judge (OpenAI-compatible endpoint)."""

    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def load_model(self) -> OpenAI:
        return self._client

    def generate(self, prompt: str, schema: Any = None, **kwargs: Any):
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"} if schema is not None else NOT_GIVEN,
        )
        content = response.choices[0].message.content or ""
        if schema is None:
            return content
        # Strip accidental markdown fences before validating.
        cleaned = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return schema.model_validate_json(cleaned)
        except ValueError:
            # One retry with an explicit instruction, then surface the error.
            retry = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Return ONLY valid JSON matching the requested schema."},
                ],
                response_format={"type": "json_object"},
            )
            return schema.model_validate_json(retry.choices[0].message.content or "")

    async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self) -> str:
        return f"deepseek/{self._model}"


@pytest.fixture(scope="module")
def query_pipeline():
    return build_query_pipeline()


@pytest.fixture(scope="module")
def judge():
    settings = get_settings()
    return DeepSeekJudge(
        api_key=settings.deepseek_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )


def _golden_rows() -> list[tuple[str, str]]:
    with GOLDEN_SET.open(encoding="utf-8") as handle:
        return [(row["question"], row["expected_answer"]) for row in csv.DictReader(handle)]


@pytest.mark.parametrize("question,expected", _golden_rows(), ids=lambda value: value[:48])
def test_golden_set(question: str, expected: str, query_pipeline, judge: DeepSeekJudge):
    answer, documents = ask(query_pipeline, question)
    retrieval_context = [document.content or "" for document in documents]

    test_case = LLMTestCase(
        input=question,
        actual_output=answer.data or "",
        expected_output=expected,
        retrieval_context=retrieval_context,
    )
    metrics = [
        FaithfulnessMetric(threshold=0.85, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=0.75, model=judge, include_reason=True),
        ContextualRecallMetric(threshold=0.70, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=0.70, model=judge),
    ]
    assert_test(test_case, metrics)
