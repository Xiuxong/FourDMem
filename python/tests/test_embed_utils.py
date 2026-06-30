"""Tests for cognition.embed_utils — unified embedding helpers."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

# Mock the embedder globally to avoid loading the real model
MOCK_VEC = [0.1] * 768


def _mock_get_embedder():
    emb = MagicMock()
    emb.embed.return_value = MOCK_VEC
    return emb


def test_ensure_embedding_returns_vector():
    """ensure_embedding should return a float list."""
    with patch("cognition.embed_utils.get_embedder", _mock_get_embedder):
        # Reset cached embedder
        import cognition.embed_utils as eu
        eu._embedder = None
        vec = eu.ensure_embedding("test text")
        assert isinstance(vec, list)
        assert all(isinstance(x, float) for x in vec)
        assert len(vec) == 768


def test_ensure_embedding_zero_on_failure():
    """ensure_embedding returns zero vector if embedder fails."""
    with patch("cognition.embed_utils.get_embedder", side_effect=RuntimeError("no model")):
        import cognition.embed_utils as eu
        eu._embedder = None
        vec = eu.ensure_embedding("test")
        assert len(vec) == 768
        assert all(x == 0.0 for x in vec)


def test_add_fact_safely_calls_engine():
    """add_fact_safely should call engine.add_fact_with_embedding."""
    with patch("cognition.embed_utils.get_embedder", _mock_get_embedder):
        import cognition.embed_utils as eu
        eu._embedder = None
        engine = MagicMock()
        engine.add_fact_with_embedding.return_value = '{"status": "fact_added"}'

        result = eu.add_fact_safely(engine, "test fact", [1, 2])
        engine.add_fact_with_embedding.assert_called_once()
        args = engine.add_fact_with_embedding.call_args
        assert args[0][0] == "test fact"
        assert args[0][1] == [1, 2]
        assert len(args[0][2]) == 768


def test_add_fact_safely_none_l0_refs():
    """add_fact_safely should pass None l0_refs through."""
    with patch("cognition.embed_utils.get_embedder", _mock_get_embedder):
        import cognition.embed_utils as eu
        eu._embedder = None
        engine = MagicMock()
        engine.add_fact_with_embedding.return_value = '{"status": "fact_added"}'

        eu.add_fact_safely(engine, "test fact", None)
        args = engine.add_fact_with_embedding.call_args
        assert args[0][1] is None


def test_ingest_safely_calls_engine():
    """ingest_safely should call engine.save_with_embedding."""
    with patch("cognition.embed_utils.get_embedder", _mock_get_embedder):
        import cognition.embed_utils as eu
        eu._embedder = None
        engine = MagicMock()
        engine.save_with_embedding.return_value = '{"status": "saved"}'

        result = eu.ingest_safely(engine, "sess-1", "user", "hello", '{"key": 1}')
        engine.save_with_embedding.assert_called_once()
        args = engine.save_with_embedding.call_args
        assert args[0][0] == "sess-1"
        assert args[0][1] == "user"
        assert args[0][2] == "hello"
        assert len(args[0][3]) == 768
        assert args[0][4] == '{"key": 1}'


def test_add_fact_safely_returns_str():
    """add_fact_safely should return str (JSON from Rust)."""
    with patch("cognition.embed_utils.get_embedder", _mock_get_embedder):
        import cognition.embed_utils as eu
        eu._embedder = None
        engine = MagicMock()
        engine.add_fact_with_embedding.return_value = '{"status": "fact_added"}'

        result = eu.add_fact_safely(engine, "test")
        assert isinstance(result, str)


if __name__ == "__main__":
    test_ensure_embedding_returns_vector()
    test_ensure_embedding_zero_on_failure()
    test_add_fact_safely_calls_engine()
    test_add_fact_safely_none_l0_refs()
    test_ingest_safely_calls_engine()
    test_add_fact_safely_returns_str()
    print("All embed_utils tests passed!")
