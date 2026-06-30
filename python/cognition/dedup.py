"""Semantic Deduplication Engine — prevents duplicate facts in L1 graph.

Before adding a new fact, searches for similar existing facts using
embedding cosine similarity (via engine.vector_search):

- similarity > 0.92 → MERGE (update tick + increment frequency)
- similarity 0.75-0.92 → LINK (create similar_to edge, still add)
- similarity < 0.75 → ADD (new fact, normal flow)
"""

import json
from typing import Any


# ── Similarity thresholds ─────────────────────────────────────────────────────

MERGE_THRESHOLD = 0.92    # Above this: merge with existing fact
LINK_THRESHOLD = 0.75     # Above this: add but link to similar


class SemanticDeduplicator:
    """Prevents duplicate facts by checking embedding cosine similarity."""

    def __init__(self, merge_threshold: float = MERGE_THRESHOLD,
                 link_threshold: float = LINK_THRESHOLD):
        self.merge_threshold = merge_threshold
        self.link_threshold = link_threshold

    def add_fact_with_dedup(
        self, engine: Any, label: str, l0_refs: list[int] | None = None,
        embedder: Any = None
    ) -> dict:
        """Add a fact with semantic deduplication.

        Uses embedding cosine similarity (via engine.vector_search) to find
        the most similar existing L1 fact, then decides: merge, link, or add.

        Args:
            engine: FourDMemEngine instance.
            label: The fact text to add.
            l0_refs: Optional L0 evidence references.
            embedder: Optional bge embedder for semantic similarity.

        Returns:
            Dict with status ("added", "merged", "linked") and details.
        """
        if not label or len(label.strip()) < 5:
            return {"status": "skipped", "reason": "too_short", "label": label}

        # Step 1: Find most similar L1 fact via embedding cosine similarity
        similar = self._find_similar(engine, label, embedder=embedder)

        # Step 2: Decide action based on true cosine similarity
        if similar and similar["score"] >= self.merge_threshold:
            return self._merge(engine, label, similar, l0_refs)
        elif similar and similar["score"] >= self.link_threshold:
            return self._link(engine, label, similar, l0_refs)
        else:
            return self._add(engine, label, l0_refs)

    def _find_similar(self, engine: Any, label: str, embedder: Any = None) -> dict | None:
        """Find the most similar L1 fact using embedding cosine similarity.

        Uses engine.vector_search() which returns true cosine similarity,
        NOT the RRF fusion score from engine.query().

        Returns the best match with its cosine similarity, or None.
        """
        if embedder is None:
            # Try to load embedder
            try:
                from cognition.embedder import get_embedder
                embedder = get_embedder()
            except ImportError:
                pass

        if embedder is None or not getattr(embedder, '_loaded', False):
            # Fallback: use engine.query (less accurate but always available)
            return self._find_similar_fallback(engine, label)

        try:
            # Get embedding for the new fact
            vec = embedder.embed(label)
            if not vec or all(x == 0.0 for x in vec):
                return self._find_similar_fallback(engine, label)

            # Pure vector search: returns true cosine similarity
            results = engine.vector_search(vec, 5)

            if not results:
                return None

            # Find the best match (skip exact duplicates)
            for node_idx, content, similarity in results:
                if content.strip().lower() != label.strip().lower():
                    return {
                        "id": str(node_idx),
                        "content": content,
                        "score": similarity,  # This is TRUE cosine similarity
                        "layer": "L1",
                    }

            return None

        except Exception:
            return self._find_similar_fallback(engine, label)

    def _find_similar_fallback(self, engine: Any, label: str) -> dict | None:
        """Fallback: use engine.query when embedder is not available.

        Note: scores from engine.query are RRF fusion scores, NOT true
        similarity. This is less accurate but always works.
        """
        try:
            raw = engine.query(label, 5)
            data = json.loads(raw) if isinstance(raw, str) else raw
            results = data.get("results", [])

            for item in results:
                layer = item.get("layer", "")
                content = item.get("content", "")
                score = item.get("score", 0.0)

                if layer in ("L1", 1, "1") and content.strip().lower() != label.strip().lower():
                    return {
                        "id": item.get("id"),
                        "content": content,
                        "score": score,
                        "layer": "L1",
                    }

            # If no L1 match, consider high-scoring results
            if results:
                best = results[0]
                if best.get("score", 0) >= self.link_threshold:
                    return {
                        "id": best.get("id"),
                        "content": best.get("content", ""),
                        "score": best.get("score", 0),
                        "layer": best.get("layer", "?"),
                    }

        except Exception:
            pass

        return None

    def _merge(self, engine: Any, label: str, similar: dict,
               l0_refs: list[int] | None) -> dict:
        """Merge with existing fact: boost utility + bump tick."""
        try:
            entity_id = similar.get("content", "")
            engine.feedback(entity_id, 0.5)
            return {
                "status": "merged",
                "label": label[:200],
                "merged_with": {
                    "id": similar["id"],
                    "content": similar["content"][:200],
                    "similarity": round(similar["score"], 3),
                },
            }
        except Exception:
            return self._add(engine, label, l0_refs)

    def _link(self, engine: Any, label: str, similar: dict,
              l0_refs: list[int] | None) -> dict:
        """Add new fact but record link to similar existing fact."""
        try:
            result = self._add(engine, label, l0_refs)
            result["linked_to"] = {
                "id": similar["id"],
                "content": similar["content"][:200],
                "similarity": round(similar["score"], 3),
            }
            result["status"] = "linked"
            return result
        except Exception:
            return self._add(engine, label, l0_refs)

    def _add(self, engine: Any, label: str,
             l0_refs: list[int] | None) -> dict:
        """Normal add: new fact with semantic embedding."""
        try:
            from cognition.embed_utils import add_fact_safely
            raw = add_fact_safely(engine, label, l0_refs)
            node_index = None
            try:
                node_index = json.loads(raw).get("node_index")
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
            result = {
                "status": "added",
                "label": label[:200],
            }
            if node_index is not None:
                result["node_index"] = node_index
            return result
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "label": label[:200],
            }
