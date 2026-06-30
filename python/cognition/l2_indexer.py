"""L2 Dual-Channel Index — Episodic + Semantic memory retrieval.

Based on:
- "Episodic-Semantic Memory Architecture for Long-Horizon Scientific Agents"
  (May 2026, arXiv)
- "TraceMem: Weaving Narrative Memory Schemata from User Conversational Traces"
  (Feb 2026, arXiv)
- "TiMem: Temporal-Hierarchical Memory Consolidation" (Jan 2026, arXiv)

Architecture:
    L2 Scenario Store
    ├── Semantic Channel (语义层)
    │   Key = "how to configure rust async runtime"
    │   Index = dense vector (bge-base-en-v1.5 768-dim)
    │   → fast concept-level retrieval
    │
    └── Episodic Channel (情景层)
        Key = "2026-06-20: tokio panic debugging session"
        Index = sparse (BM25) + dense hybrid
        → narrative timeline retrieval
"""

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ── Narrative Schema (TraceMem) ──────────────────────────────────────────────

class NarrativeSchema:
    """TraceMem-style narrative schema extracted from conversations.

    Fields follow TraceMem's taxonomy:
      - trigger_event: what started this interaction
      - context: project/task context
      - action: what the agent did
      - outcome: result of the action
      - lesson: extracted knowledge
    """

    __slots__ = (
        "id", "trigger_event", "context", "action", "outcome",
        "lesson", "workspace_id", "session_id", "timestamp", "tick",
    )

    def __init__(
        self,
        trigger_event: str = "",
        context: str = "",
        action: str = "",
        outcome: str = "",
        lesson: str = "",
        workspace_id: str = "default",
        session_id: str = "",
        tick: int = 0,
    ):
        self.id = hashlib.md5(
            f"{trigger_event}:{action}:{lesson}".encode()
        ).hexdigest()[:12]
        self.trigger_event = trigger_event
        self.context = context
        self.action = action
        self.outcome = outcome
        self.lesson = lesson
        self.workspace_id = workspace_id
        self.session_id = session_id
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.tick = tick

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "trigger_event": self.trigger_event,
            "context": self.context,
            "action": self.action,
            "outcome": self.outcome,
            "lesson": self.lesson,
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "tick": self.tick,
        }

    def to_text(self) -> str:
        """Convert schema to a searchable text representation."""
        return (
            f"Trigger: {self.trigger_event}\n"
            f"Context: {self.context}\n"
            f"Action: {self.action}\n"
            f"Outcome: {self.outcome}\n"
            f"Lesson: {self.lesson}"
        )


# ── Dual Index ────────────────────────────────────────────────────────────────

class L2DualChannelIndex:
    """Two-channel L2 scenario index.

    Semantic channel: dense embeddings for concept search.
    Episodic channel: narrative schemata for timeline/context search.
    """

    def __init__(self, vault_root: str = ""):
        self.vault_root = vault_root or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "vault",
        )
        self.scenarios_dir = os.path.join(self.vault_root, "scenarios")
        os.makedirs(self.scenarios_dir, exist_ok=True)

        # Semantic channel: FAISS flat index (dense)
        self._semantic_index = None          # faiss.IndexFlatIP
        self._semantic_ids: list[str] = []   # doc_id per vector
        self._semantic_embeddings: list[np.ndarray] = []
        self._dim = 768  # bge-base-en-v1.5 output dim

        # Episodic channel: schemata store
        self._schemata: dict[str, NarrativeSchema] = {}

        # Embedder
        self._embedder = None

    # ── Embedder ──────────────────────────────────────────────────────────

    def _get_embedder(self):
        """Lazy-load the embedder."""
        if self._embedder is None:
            try:
                from cognition.embedder import get_embedder as _get
                self._embedder = _get()
            except Exception:
                self._embedder = None
        return self._embedder

    def _embed(self, text: str) -> list[float]:
        """Compute dense embedding."""
        emb_model = self._get_embedder()
        if emb_model is not None:
            return emb_model.embed(text)
        # Fallback: simple bag-of-words hash
        return [0.0] * self._dim

    # ── Semantic Channel ──────────────────────────────────────────────────

    def index_semantic(self, scenario_id: str, content: str):
        """Add a scenario to the semantic (dense) channel."""
        try:
            import faiss
        except ImportError:
            logger.warning("faiss not installed, semantic channel disabled")
            return

        embedding = self._embed(content)
        emb_np = np.array(embedding, dtype=np.float32).reshape(1, -1)

        if self._semantic_index is None:
            self._semantic_index = faiss.IndexFlatIP(self._dim)

        self._semantic_index.add(emb_np)
        self._semantic_ids.append(scenario_id)
        self._semantic_embeddings.append(emb_np.flatten())

        logger.debug(f"Semantic indexed: {scenario_id} ({len(content)} chars)")

    def search_semantic(self, query: str, top_k: int = 5) -> list[dict]:
        """Search semantic channel by concept similarity."""
        if self._semantic_index is None or self._semantic_index.ntotal == 0:
            return []

        embedding = self._embed(query)
        emb_np = np.array(embedding, dtype=np.float32).reshape(1, -1)

        scores, indices = self._semantic_index.search(emb_np, min(top_k, self._semantic_index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._semantic_ids):
                continue
            results.append({
                "channel": "semantic",
                "scenario_id": self._semantic_ids[idx],
                "score": float(score),
            })

        return results

    # ── Episodic Channel ──────────────────────────────────────────────────

    def add_schema(self, schema: NarrativeSchema):
        """Add a narrative schema to the episodic channel."""
        self._schemata[schema.id] = schema
        logger.debug(f"Episodic schema added: {schema.id} — {schema.trigger_event[:50]}")

    def search_episodic(
        self, query: str, top_k: int = 5
    ) -> list[dict]:
        """Search episodic channel by keyword + embedding hybrid."""
        results = []

        for schema_id, schema in self._schemata.items():
            text = schema.to_text()
            # BM25-like sparse scoring
            bm25_score = self._bm25_score(query, text)
            # Dense semantic similarity
            dense_score = self._cosine_sim(self._embed(query), self._embed(text))
            # Hybrid: 0.4 * BM25 + 0.6 * dense
            hybrid = 0.4 * bm25_score + 0.6 * dense_score

            if hybrid > 0.1:
                results.append({
                    "channel": "episodic",
                    "schema_id": schema_id,
                    "trigger_event": schema.trigger_event,
                    "lesson": schema.lesson,
                    "score": hybrid,
                    "timestamp": schema.timestamp,
                    "workspace_id": schema.workspace_id,
                })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    # ── TraceMem Narrative Extraction ─────────────────────────────────────

    def extract_schema_from_turn(
        self,
        user_message: str,
        assistant_response: str,
        workspace_id: str = "default",
        session_id: str = "",
        tick: int = 0,
    ) -> Optional[NarrativeSchema]:
        """TraceMem: extract narrative schema from a conversation turn.

        Uses keyword heuristics + LLM fallback to identify:
          1. trigger_event — problem/question being addressed
          2. context — project/task context
          3. action — what was done
          4. outcome — result/answer
          5. lesson — extractable knowledge
        """
        combined = f"{user_message}\n{assistant_response}"

        # Heuristic extraction
        trigger = self._extract_trigger(user_message)
        context = self._extract_context(user_message)
        action = self._extract_action(assistant_response)
        outcome = self._extract_outcome(assistant_response)
        lesson = self._extract_lesson(combined)

        if not trigger:
            return None

        schema = NarrativeSchema(
            trigger_event=trigger,
            context=context,
            action=action,
            outcome=outcome,
            lesson=lesson,
            workspace_id=workspace_id,
            session_id=session_id,
            tick=tick,
        )

        self.add_schema(schema)
        return schema

    @staticmethod
    def _extract_trigger(text: str) -> str:
        """Extract the trigger event from user message."""
        # Pattern: question marks, error descriptions, requests
        patterns = [
            r'(?:error|bug|issue|problem|失败|报错|异常)[：:]\s*(.+?)(?:[。\n]|$)',
            r'(?:报错信息|错误提示)[：:]\s*(.+?)(?:[。\n]|$)',
            r'^(.+?)[？?]',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()[:200]
        return text[:200]

    @staticmethod
    def _extract_context(text: str) -> str:
        """Extract project/task context."""
        patterns = [
            r'(?:项目|project|在|On)\s*(.+?)(?:[，,]\s*.+?(?:报错|失败|问题))',
            r'(?:context|背景)[：:]\s*(.+?)(?:[。\n]|$)',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()[:200]
        return ""

    @staticmethod
    def _extract_action(text: str) -> str:
        """Extract what the agent did."""
        action_markers = [
            r'(?:修改|fix|修复|change|update|add|create|remove)[：:]\s*(.+?)(?:[。\n]|$)',
            r'(?:方案|solution|plan)[：:]\s*(.+?)(?:[。\n]|$)',
        ]
        for pat in action_markers:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()[:300]
        # Fallback: first meaningful line of assistant response
        for line in text.split("\n"):
            line = line.strip()
            if len(line) > 20 and not line.startswith("#"):
                return line[:300]
        return ""

    @staticmethod
    def _extract_outcome(text: str) -> str:
        """Extract the outcome."""
        patterns = [
            r'(?:结果|result|outcome|✅|✓)[：:]\s*(.+?)(?:[。\n]|$)',
            r'(?:测试|test)[：:]\s*(.+?)(?:[。\n]|$)',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()[:200]
        return ""

    @staticmethod
    def _extract_lesson(text: str) -> str:
        """Extract the lesson / takeaway knowledge."""
        patterns = [
            r'(?:关键|crucial|lesson|takeaway|总结|教训|要点)[：:]\s*(.+?)(?:[。\n]|$)',
            r'(?:根本原因|root cause)[：:]\s*(.+?)(?:[。\n]|$)',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()[:300]
        return ""

    # ── Index all scenarios on disk ───────────────────────────────────────

    def index_disk_scenarios(self):
        """Index all existing markdown scenarios in the vault."""
        if not os.path.exists(self.scenarios_dir):
            return

        for md_file in Path(self.scenarios_dir).glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                scenario_id = md_file.stem
                # Index in semantic channel
                self.index_semantic(scenario_id, content)
                logger.info(f"Indexed scenario: {scenario_id}")
            except Exception as e:
                logger.warning(f"Failed to index {md_file}: {e}")

    def serialize_schemata(self, path: str):
        """Persist schemata to JSON for recovery."""
        data = {sid: s.to_dict() for sid, s in self._schemata.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_schemata(self, path: str):
        """Load schemata from JSON."""
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sid, sdict in data.items():
            schema = NarrativeSchema(
                trigger_event=sdict.get("trigger_event", ""),
                context=sdict.get("context", ""),
                action=sdict.get("action", ""),
                outcome=sdict.get("outcome", ""),
                lesson=sdict.get("lesson", ""),
                workspace_id=sdict.get("workspace_id", "default"),
                session_id=sdict.get("session_id", ""),
                tick=sdict.get("tick", 0),
            )
            schema.id = sid
            self._schemata[sid] = schema

    # ── Search API (unified) ──────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> dict:
        """Unified two-channel search.

        Returns combined results from semantic + episodic channels.
        """
        semantic = self.search_semantic(query, top_k)
        episodic = self.search_episodic(query, top_k)

        # Normalize and merge scores
        all_results = self._merge_channels(semantic, episodic, top_k)
        return {
            "query": query,
            "semantic_count": len(semantic),
            "episodic_count": len(episodic),
            "results": all_results,
        }

    def _merge_channels(
        self, semantic: list[dict], episodic: list[dict], top_k: int
    ) -> list[dict]:
        """Merge semantic and episodic results with score normalization."""
        merged = []

        # Semantic results: score range depends on vector index, normalize to [0,1]
        sem_max = max((r["score"] for r in semantic), default=1.0)
        for r in semantic:
            r["score"] = r["score"] / max(sem_max, 0.001)
            r["weight"] = 0.6  # semantic channel weight
            merged.append(r)

        # Episodic results: already in [0,1] range
        for r in episodic:
            r["weight"] = 0.4  # episodic channel weight
            merged.append(r)

        # Sort by weighted score
        merged.sort(
            key=lambda r: r["score"] * r.get("weight", 0.5),
            reverse=True,
        )
        return merged[:top_k]

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _bm25_score(query: str, doc: str, k1: float = 1.2, b: float = 0.75) -> float:
        """Simple BM25 scoring without IDF (single-document context)."""
        query_terms = query.lower().split()
        doc_terms = doc.lower().split()
        doc_len = len(doc_terms)
        avg_dl = max(doc_len, 1)

        score = 0.0
        for term in query_terms:
            tf = doc_terms.count(term)
            if tf == 0:
                continue
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_dl)
            score += numerator / max(denominator, 0.001)

        return min(score, 10.0) / 10.0  # normalize to [0,1]

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Cosine similarity."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get_stats(self) -> dict:
        """Return index statistics."""
        return {
            "semantic_docs": self._semantic_index.ntotal if self._semantic_index else 0,
            "episodic_schemata": len(self._schemata),
            "embedding_dim": self._dim,
        }


# ── Singleton factory ─────────────────────────────────────────────────────────

_l2_index: Optional[L2DualChannelIndex] = None


def get_l2_index(vault_root: str = "") -> L2DualChannelIndex:
    """Lazy singleton for L2DualChannelIndex."""
    global _l2_index
    if _l2_index is None:
        _l2_index = L2DualChannelIndex(vault_root=vault_root)
        # Auto-index disk scenarios on first load
        _l2_index.index_disk_scenarios()
    return _l2_index
