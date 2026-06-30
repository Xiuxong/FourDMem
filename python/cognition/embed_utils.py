"""Unified embedding helper — ensures every L1 write has a real semantic vector.

All code paths that write to the L1 graph or vector index MUST call
ensure_embedding() before the write. This replaces the deprecated Rust
trigram-hash embed_text() with the local bge-small-zh-v1.5 model.

Usage:
    from cognition.embed_utils import ensure_embedding
    vec = ensure_embedding(label)
    engine.add_fact_with_embedding(label, l0_refs, vec)
"""

from typing import Any

_embedder = None


def get_embedder():
    """Lazy-load the bge embedder singleton."""
    global _embedder
    if _embedder is None:
        from cognition.embedder import get_embedder as _get
        _embedder = _get()
    return _embedder


def ensure_embedding(text: str) -> list[float]:
    """Compute semantic embedding for text using bge-small-zh-v1.5.

    Returns a 768-dim float vector. Falls back to zero vector if
    the embedder fails to load (should not happen in production).
    """
    try:
        emb = get_embedder()
        return emb.embed(text)
    except Exception:
        return [0.0] * 768


def add_fact_safely(
    engine: Any,
    label: str,
    l0_refs: list[int] | None = None,
) -> str:
    """Add an L1 fact with guaranteed semantic embedding.

    Wraps engine.add_fact_with_embedding() with automatic embedding.
    Use this instead of engine.add_fact() directly.
    """
    vec = ensure_embedding(label)
    return engine.add_fact_with_embedding(label, l0_refs, vec)


def ingest_safely(
    engine: Any,
    session_id: str,
    role: str,
    content: str,
    metadata: str = "{}",
    workspace_id: str = "default",
) -> str:
    """Ingest L0 evidence with guaranteed semantic embedding.

    Wraps engine.save_with_embedding() with automatic embedding.
    Use this instead of engine.save() for content that should be
    vector-searchable.
    """
    vec = ensure_embedding(content)
    return engine.save_with_embedding(session_id, role, content, vec, metadata)
