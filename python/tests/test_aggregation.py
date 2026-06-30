"""Tests for cognition.aggregator — Weighted L1→L2 auto-aggregation."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cognition.aggregator import AutoAggregator


def test_record_fact_below_threshold():
    """Facts below threshold should not trigger aggregation."""
    agg = AutoAggregator(aggregation_threshold=5)
    result = agg.record_fact({"label": "rust borrow checker", "node_index": 1})
    assert result is None


def test_record_fact_triggers_aggregation_weighted():
    """Conversation facts (weight 2.0) should trigger faster than tool facts."""
    agg = AutoAggregator(aggregation_threshold=5)
    # 3 conversation facts = 3 * 2.0 = 6.0 >= 5.0 → triggers
    for i in range(3):
        result = agg.record_fact(
            {"label": f"rust memory safety fact {i}", "node_index": i},
            source="conversation"
        )
    assert result is not None
    assert result["status"] == "aggregated"
    assert result["fact_count"] == 3


def test_record_fact_tool_needs_more():
    """Tool facts (weight 1.0) need more to trigger."""
    agg = AutoAggregator(aggregation_threshold=5)
    # 4 tool facts = 4 * 1.0 = 4.0 < 5.0 → no trigger
    for i in range(4):
        result = agg.record_fact(
            {"label": f"cargo build error {i}", "node_index": i},
            source="tool"
        )
    assert result is None
    # 5th tool fact = 5.0 >= 5.0 → triggers
    result = agg.record_fact(
        {"label": "cargo build error 5", "node_index": 5},
        source="tool"
    )
    assert result is not None


def test_record_fact_mixed_sources():
    """Mix of conversation and tool facts should aggregate correctly."""
    agg = AutoAggregator(aggregation_threshold=5)
    # 2 conversation (4.0) + 1 tool (1.0) = 5.0 → triggers
    agg.record_fact({"label": "SQLite 存储方案确定", "node_index": 0}, source="conversation")
    agg.record_fact({"label": "SQLite 性能优于预期", "node_index": 1}, source="conversation")
    result = agg.record_fact({"label": "SQLite 连接测试成功", "node_index": 2}, source="tool")
    assert result is not None
    assert result["status"] == "aggregated"


def test_aggregation_extracts_topic_english():
    """Aggregation should extract English topic from fact labels."""
    agg = AutoAggregator(aggregation_threshold=2)
    agg.record_fact({"label": "database migration strategy is critical", "node_index": 1})
    result = agg.record_fact({"label": "database migration strategy needs planning", "node_index": 2})
    assert result is not None
    assert "database" in result["topic"].lower()


def test_aggregation_resets_cluster():
    """After aggregation, the cluster should be reset."""
    agg = AutoAggregator(aggregation_threshold=2)
    agg.record_fact({"label": "test fact alpha version one", "node_index": 1})
    agg.record_fact({"label": "test fact alpha version two", "node_index": 2})
    # Cluster should be reset after aggregation
    assert len(agg.get_pending_clusters()) == 0


def test_get_pending_clusters():
    """Pending clusters should show weighted score info."""
    agg = AutoAggregator(aggregation_threshold=5)
    agg.record_fact({"label": "test fact one", "node_index": 1})
    agg.record_fact({"label": "test fact two", "node_index": 2})
    pending = agg.get_pending_clusters()
    assert len(pending) > 0
    for topic, info in pending.items():
        assert "count" in info
        assert "remaining" in info
        assert "weighted_score" in info


def test_get_stats():
    """get_stats should return cluster counts."""
    agg = AutoAggregator(aggregation_threshold=5)
    agg.record_fact({"label": "alpha beta gamma", "node_index": 1})
    stats = agg.get_stats()
    assert "active_clusters" in stats
    assert "total_pending_facts" in stats


def test_force_aggregate():
    """force_aggregate should query engine and aggregate."""
    from unittest.mock import MagicMock
    import json
    agg = AutoAggregator(aggregation_threshold=2)
    engine = MagicMock()
    engine.query.return_value = json.dumps({
        "results": [
            {"layer": "L1", "content": "fact 1", "id": 1},
            {"layer": "L1", "content": "fact 2", "id": 2},
        ]
    })
    result = agg.force_aggregate(engine, "test topic")
    assert result["status"] == "aggregated"


if __name__ == "__main__":
    test_record_fact_below_threshold()
    test_record_fact_triggers_aggregation_weighted()
    test_record_fact_tool_needs_more()
    test_record_fact_mixed_sources()
    test_aggregation_extracts_topic_chinese()
    test_aggregation_extracts_topic_english()
    test_aggregation_resets_cluster()
    test_get_pending_clusters()
    test_get_stats()
    test_force_aggregate()
    print("All aggregation tests passed!")
