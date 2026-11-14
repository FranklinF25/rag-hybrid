"""DeepEval harness: RAG metrics over the Go rag-hybrid HTTP API.

Language-agnostic since the Go rewrite: this suite only talks to the
running API, so it can evaluate any implementation of the contract.

Run:
    pytest -m eval evals/          (with the webapp serving on RAG_URL)

Prerequisites (the suite skips with a reason otherwise):
- the rag-hybrid API reachable (RAG_URL, default http://127.0.0.1:8000)
- ``DEEPSEEK_API_KEY`` for the judge (env or .env)

Thresholds follow the research baseline: faithfulness >= 0.85 and
contextual precision >= 0.75 are pinned; recall and relevancy use 0.70.
"""

from __future__ import annotations

import csv
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import pytest

GOLDEN_SET = Path(__file__).parent / "golden_set.csv"
RAG_URL = os.environ.get("RAG_URL", "http://127.0.0.1:8000").rstrip("/")
HEALTH_TIMEOUT = 3
QUERY_TIMEOUT = 180

pytestmark = pytest.mark.eval


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE, never overrides real env vars)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


def api_get(path: str, timeout: float = QUERY_TIMEOUT) -> dict[str, Any]:
    with urllib.request.urlopen(f"{RAG_URL}{path}", timeout=timeout) as response:
        return json.load(response)


def api_chat(message: str) -> dict[str, Any]:
    payload = json.dumps({"message": message}).encode("utf-8")
    request = urllib.request.Request(
        f"{RAG_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=QUERY_TIMEOUT) as response:
        return json.load(response)


def _prerequisite_reason() -> str | None:
    try:
        api_get("/api/health", timeout=HEALTH_TIMEOUT)
    except OSError:
        return f"rag-hybrid API not reachable at {RAG_URL} (start it: rag-hybrid serve / docker compose up webapp)"
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return "DEEPSEEK_API_KEY is not set (judge)"
    return None


_PREREQUISITE_REASON = _prerequisite_reason()
if _PREREQUISITE_REASON:
    pytest.skip(_PREREQUISITE_REASON, allow_module_level=True)


class DeepSeekJudge:  # DeepEvalBaseLLM, imported lazily after the skip gate
    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        from openai import OpenAI

        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def load_model(self):
        return self._client

    def generate(self, prompt: str, schema: Any = None, **kwargs: Any):
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )
        content = response.choices[0].message.content or ""
        if schema is None:
            return content
        cleaned = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return schema.model_validate_json(cleaned)

    async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self) -> str:
        return f"deepseek/{self._model}"


@pytest.fixture(scope="module")
def judge():
    from deepeval.models import DeepEvalBaseLLM

    # Dynamically attach the base class so the import stays behind the gate.
    return type("Judge", (DeepSeekJudge, DeepEvalBaseLLM), {})(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        model=os.environ.get("LLM_MODEL", "deepseek-chat"),
    )


def _golden_rows() -> list[tuple[str, str]]:
    with GOLDEN_SET.open(encoding="utf-8") as handle:
        return [(row["question"], row["expected_answer"]) for row in csv.DictReader(handle)]


@pytest.mark.parametrize("question,expected", _golden_rows(), ids=lambda value: value[:48])
def test_golden_set(question: str, expected: str, judge):
    from deepeval import assert_test
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    retrieval = api_get(f"/api/search?q={urllib.request.quote(question)}&full=1")
    contexts = [result["content"] for result in retrieval["results"]]
    chat = api_chat(question)

    test_case = LLMTestCase(
        input=question,
        actual_output=chat["answer"],
        expected_output=expected,
        retrieval_context=contexts,
    )
    metrics = [
        FaithfulnessMetric(threshold=0.85, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=0.75, model=judge, include_reason=True),
        ContextualRecallMetric(threshold=0.70, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=0.70, model=judge),
    ]
    assert_test(test_case, metrics)
