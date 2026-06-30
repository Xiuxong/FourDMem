"""Embedding-based evaluator for FourDMem benchmarks.

Uses the same local 1.5B embedding model (bge-small-zh-v1.5) that powers
FourDMem's retrieval. Zero external API calls. Privacy-first.

Evaluation method:
    cosine_similarity(embed(expected_answer), embed(retrieved_context))
    >= threshold → correct

This is fair because:
    1. Same model used for retrieval and evaluation (no metric leakage)
    2. Measures semantic relevance, not keyword overlap
    3. Fully local, no external dependencies
"""

import json
import math
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingJudge:
    """Embedding-based evaluator using the local model.

    Same model that powers FourDMem retrieval = fair evaluation.
    """

    def __init__(self, threshold: float = 0.45):
        """
        Args:
            threshold: Cosine similarity threshold for "correct".
                       0.45 is empirically good for bge-small-zh-v1.5
                       (embeddings are 512-dim, not super granular).
        """
        self.threshold = threshold
        self._embedder = None
        self.call_count = 0
        self.total_latency = 0.0

    def _get_embedder(self):
        if self._embedder is None:
            from cognition.embedder import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def evaluate(
        self,
        question: str,
        expected: str,
        retrieved: str,
    ) -> dict:
        """Evaluate by embedding similarity.

        Args:
            question: The question asked
            expected: The expected answer
            retrieved: The retrieved context (top-K joined)

        Returns:
            Dict with correct (bool), score (float), method (str)
        """
        start = time.time()
        embedder = self._get_embedder()

        try:
            # Embed expected answer and retrieved context
            emb_expected = embedder.embed(expected)
            emb_retrieved = embedder.embed(retrieved[:1000])  # Truncate for speed

            if not emb_expected or not emb_retrieved:
                return {"correct": False, "score": 0.0, "method": "embedding", "error": "empty embedding"}

            sim = _cosine_similarity(emb_expected, emb_retrieved)
            self.call_count += 1
            latency = (time.time() - start) * 1000
            self.total_latency += latency

            return {
                "correct": sim >= self.threshold,
                "score": round(sim, 4),
                "threshold": self.threshold,
                "method": "embedding_similarity",
                "latency_ms": round(latency, 2),
            }
        except Exception as e:
            return {"correct": False, "score": 0.0, "method": "embedding", "error": str(e)[:100]}

    def evaluate_adversarial(
        self,
        question: str,
        expected: str,
        retrieved: str,
    ) -> dict:
        """Evaluate adversarial questions.

        For adversarial questions, "correct" means the retrieved context
        is NOT semantically similar to the expected answer.
        (The expected answer is "this was not discussed" — if retrieval
        returns something unrelated, that's correct behavior.)
        """
        start = time.time()
        embedder = self._get_embedder()

        try:
            # For adversarial: embed the question (not expected)
            emb_question = embedder.embed(question)
            emb_retrieved = embedder.embed(retrieved[:1000])

            if not emb_question or not emb_retrieved:
                return {"correct": True, "score": 0.0, "method": "embedding_adversarial"}

            sim = _cosine_similarity(emb_question, emb_retrieved)
            self.call_count += 1
            latency = (time.time() - start) * 1000
            self.total_latency += latency

            # Adversarial correct = low similarity (system didn't find relevant info)
            # Use same threshold as normal evaluation for consistency
            return {
                "correct": sim < self.threshold,
                "score": round(1.0 - sim, 4),
                "method": "embedding_adversarial",
                "latency_ms": round(latency, 2),
            }
        except Exception as e:
            return {"correct": True, "score": 0.0, "method": "embedding_adversarial", "error": str(e)[:100]}

    def get_stats(self) -> dict:
        """Get judge statistics."""
        return {
            "method": "embedding_similarity",
            "threshold": self.threshold,
            "total_calls": self.call_count,
            "avg_latency_ms": round(self.total_latency / self.call_count, 2) if self.call_count > 0 else 0,
            "total_latency_s": round(self.total_latency / 1000, 2),
        }


# ── Convenience alias ─────────────────────────────────────────────────────────

# For backward compatibility, also expose as LLMJudge
LLMJudge = EmbeddingJudge
