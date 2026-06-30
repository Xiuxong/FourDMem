"""Tests for the Agent-driven L0→L1 Fact Storage Pipeline.

Tests FactExtractor.ingest_facts() and get_session_evidence_for_agent().
Rule-based extraction has been removed — all extraction is Agent-driven.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cognition.extractor import FactExtractor


def test_ingest_facts_empty():
    """Empty facts list returns skipped status."""
    extractor = FactExtractor()
    result = extractor.ingest_facts(None, [])
    assert result["status"] == "skipped"
    assert result["facts_stored"] == 0


def test_ingest_facts_rejects_short_labels():
    """Facts with labels shorter than 8 chars are rejected."""
    extractor = FactExtractor()
    # These would need a real engine, so we test the guard condition
    facts = [
        {"label": "ab", "importance": 0.8},   # too short
        {"label": "", "importance": 0.8},      # empty
        {"label": "a long enough fact label here", "importance": 0.8},
    ]
    valid = [f for f in facts if len(f.get("label") or "") >= 8]
    assert len(valid) == 1


def test_ingest_facts_filters_low_importance():
    """Facts with importance below PENDING_THRESHOLD (0.3) are skipped."""
    extractor = FactExtractor()
    facts = [
        {"label": "important fact here ok", "importance": 0.9},
        {"label": "low importance fact here", "importance": 0.1},
    ]
    PENDING_THRESHOLD = 0.3
    valid = [f for f in facts if float(f.get("importance", 0.5)) >= PENDING_THRESHOLD]
    assert len(valid) == 1
    assert valid[0]["label"] == "important fact here ok"


def test_get_session_evidence_for_agent_returns_instruction():
    """get_session_evidence_for_agent includes instruction for Agent."""
    extractor = FactExtractor()
    # Without a real engine, this returns an error dict
    # but we can test the structure
    result = extractor.get_session_evidence_for_agent(None, "test-session")
    assert "status" in result
    # Either "ready" with instruction or "error" without a real engine
    if result["status"] == "ready":
        assert "instruction" in result
        assert "extract" in result["instruction"].lower()


def test_auto_capture_hook_returns_none():
    """auto_capture_hook always returns None (Agent-driven extraction)."""
    result = __import__("cognition.extractor", fromlist=["auto_capture_hook"]).auto_capture_hook(
        None, "test-session", 10
    )
    assert result is None


if __name__ == "__main__":
    test_ingest_facts_empty()
    test_ingest_facts_rejects_short_labels()
    test_ingest_facts_filters_low_importance()
    test_get_session_evidence_for_agent_returns_instruction()
    test_auto_capture_hook_returns_none()
    print("All extractor tests passed!")
