"""Tests for the full L0→L1→L2→L3 cognition pipeline.

Tests integration of: extractor, dedup, aggregator, salience.
Uses mocked engine — no Rust dependency, no model load.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock


def test_ingest_facts_pipeline():
    """Agent-driven ingest accepts pre-extracted facts."""
    from cognition.extractor import FactExtractor
    extractor = FactExtractor()
    engine = MagicMock()
    engine.feedback.return_value = '{"ok": true}'
    engine.add_fact_with_embedding.return_value = json.dumps({"status": "fact_added"})
    engine.query.return_value = json.dumps({"results": []})
    engine.query_with_embedding.return_value = json.dumps({"results": []})
    engine.add_fact.return_value = json.dumps({"node_index": 0, "status": "added"})

    facts = [
        {"label": "The borrow checker prevents data races at compile time", "importance": 0.9, "tags": ["rust"], "l0_refs": [1]},
        {"label": "Rust uses ownership for memory safety without garbage collection", "importance": 0.85, "tags": ["rust"], "l0_refs": [2]},
        {"label": "Lifetimes help the compiler verify reference validity", "importance": 0.8, "tags": ["rust"], "l0_refs": [3]},
    ]
    result = extractor.ingest_facts(engine, facts)
    assert "status" in result
    assert "facts_submitted" in result
    assert result["facts_submitted"] == 3


def test_get_evidence_for_agent():
    """get_session_evidence_for_agent returns evidence for Agent processing."""
    from cognition.extractor import FactExtractor
    extractor = FactExtractor()
    engine = MagicMock()
    engine.get_session_evidence.return_value = json.dumps({
        "evidence": [
            {"id": 1, "role": "user", "content": "The database migration requires careful planning and testing before deployment."},
            {"id": 2, "role": "assistant", "content": "Yes, the migration must be done in stages to avoid data loss in production."},
            {"id": 3, "role": "user", "content": "We should use the blue-green deployment strategy for safety and rollback."},
        ]
    })
    result = extractor.get_session_evidence_for_agent(engine, "test-session")
    assert result["status"] == "ready"
    assert result["evidence_count"] == 3
    assert "transcript" in result
    assert "instruction" in result
    assert "extract" in result["instruction"].lower()


def test_dedup_merge_high_similarity():
    """Dedup merges when score > 0.92."""
    from cognition.dedup import SemanticDeduplicator
    dedup = SemanticDeduplicator(merge_threshold=0.92, link_threshold=0.75)
    engine = MagicMock()
    # vector_search returns (node_idx, content, similarity) tuples — non-exact match
    engine.vector_search.return_value = [(1, "rust memory safety via borrow checker", 0.95)]
    engine.feedback.return_value = '{"ok": true}'
    # Mock embedder so _find_similar uses vector_search instead of fallback
    embedder = MagicMock()
    embedder._loaded = True
    embedder.embed.return_value = [0.1] * 768
    result = dedup.add_fact_with_dedup(engine, "rust borrow checker", embedder=embedder)
    assert result["status"] == "merged"


def test_dedup_link_medium_similarity():
    """Dedup links when 0.75 < score < 0.92."""
    from cognition.dedup import SemanticDeduplicator
    dedup = SemanticDeduplicator(merge_threshold=0.92, link_threshold=0.75)
    engine = MagicMock()
    engine.vector_search.return_value = [(1, "memory management patterns", 0.80)]
    engine.add_fact_with_embedding.return_value = json.dumps({"status": "fact_added"})
    embedder = MagicMock()
    embedder._loaded = True
    embedder.embed.return_value = [0.1] * 768
    result = dedup.add_fact_with_dedup(engine, "ownership prevents memory leaks", embedder=embedder)
    assert result["status"] == "linked"
    assert "linked_to" in result


def test_dedup_add_low_similarity():
    """Dedup adds as new when score < 0.75."""
    from cognition.dedup import SemanticDeduplicator
    dedup = SemanticDeduplicator(merge_threshold=0.92, link_threshold=0.75)
    engine = MagicMock()
    engine.vector_search.return_value = [(1, "database indexing strategies", 0.30)]
    engine.add_fact_with_embedding.return_value = json.dumps({"status": "fact_added"})

    result = dedup.add_fact_with_dedup(engine, "rust borrow checker prevents data races")
    assert result["status"] == "added"


def test_salience_detects_high_value():
    """Salience detector should flag high-value content."""
    from cognition.salience import SalienceDetector
    det = SalienceDetector(threshold=2.0)
    score = det.check("remember this critical architecture decision")
    assert score >= 2.0
    assert det.should_extract() is True


def test_salience_ignores_low_value():
    """Salience detector should ignore greetings."""
    from cognition.salience import SalienceDetector
    det = SalienceDetector(threshold=2.0)
    score = det.check("ok")
    assert score == 0.0
    assert det.should_extract() is False


def test_aggregator_triggers_at_threshold():
    """Aggregator should trigger when facts reach threshold."""
    from cognition.aggregator import AutoAggregator
    agg = AutoAggregator(aggregation_threshold=3)
    results = []
    for i in range(3):
        r = agg.record_fact({"label": f"database optimization approach {i}", "node_index": i})
        if r is not None:
            results.append(r)
    assert len(results) == 1
    assert results[0]["status"] == "aggregated"
    assert results[0]["fact_count"] == 3


def test_aggregator_resets_after_trigger():
    """Aggregator cluster should reset after aggregation."""
    from cognition.aggregator import AutoAggregator
    agg = AutoAggregator(aggregation_threshold=2)
    agg.record_fact({"label": "alpha beta gamma first", "node_index": 1})
    agg.record_fact({"label": "alpha beta gamma second", "node_index": 2})
    assert len(agg.get_pending_clusters()) == 0


def test_full_flow_extract_dedup_aggregate():
    """Full flow: Agent extracts → ingest → dedup → aggregate."""
    from cognition.extractor import FactExtractor
    from cognition.dedup import SemanticDeduplicator
    from cognition.aggregator import AutoAggregator
    from cognition.salience import SalienceDetector

    # 1. Salience check
    det = SalienceDetector(threshold=2.0)
    content = "we must remember that the database migration requires careful planning"
    score = det.check(content)
    assert score >= 2.0

    # 2. Agent pre-extracts facts (simulated)
    agent_facts = [
        {"label": "Database migration requires careful planning and testing", "importance": 0.9, "tags": ["database"], "l0_refs": [1]},
        {"label": "Migration must be done in stages to avoid data loss", "importance": 0.85, "tags": ["database"], "l0_refs": [2]},
        {"label": "Blue-green deployment provides safety and rollback during migration", "importance": 0.8, "tags": ["devops"], "l0_refs": [3]},
    ]

    # 3. Dedup (all new — no existing facts)
    dedup = SemanticDeduplicator()
    engine = MagicMock()
    engine.query.return_value = json.dumps({"results": []})
    engine.add_fact_with_embedding.return_value = json.dumps({"status": "fact_added"})
    engine.feedback.return_value = '{"ok": true}'

    added_facts = []
    for fact in agent_facts:
        result = dedup.add_fact_with_dedup(engine, fact["label"], fact.get("l0_refs"))
        if result.get("status") in ("added", "linked"):
            added_facts.append(fact)

    # 4. Ingest via FactExtractor pipeline
    extractor = FactExtractor()
    engine2 = MagicMock()
    engine2.feedback.return_value = '{"ok": true}'
    engine2.add_fact_with_embedding.return_value = json.dumps({"status": "fact_added"})
    engine2.query.return_value = json.dumps({"results": []})
    engine2.query_with_embedding.return_value = json.dumps({"results": []})
    engine2.add_fact.return_value = json.dumps({"node_index": 0, "status": "added"})

    result = extractor.ingest_facts(engine2, agent_facts)
    assert result["status"] == "ingested"
    assert result["facts_submitted"] == 3
if __name__ == "__main__":
    test_ingest_facts_pipeline()
    test_get_evidence_for_agent()
    test_dedup_merge_high_similarity()
    test_dedup_link_medium_similarity()
    test_dedup_add_low_similarity()
    test_salience_detects_high_value()
    test_salience_ignores_low_value()
    test_aggregator_triggers_at_threshold()
    test_aggregator_resets_after_trigger()
    test_full_flow_extract_dedup_aggregate()
    print("All pipeline tests passed!")
