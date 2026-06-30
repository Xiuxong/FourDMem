"""Tests for V4.0 Evolution modules: Paradigm Shift + Strange Loop."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evolution.paradigm_shift import ParadigmShiftEngine
from evolution.strange_loop import ObserverNode


def test_paradigm_shift_no_crisis():
    """No crisis when failure rate is below threshold."""
    engine = ParadigmShiftEngine(failure_threshold=0.3)
    engine.record_outcome("rust", True)
    engine.record_outcome("rust", True)
    engine.record_outcome("rust", True)

    is_crisis, rate = engine.check_crisis("rust")
    assert not is_crisis
    assert rate == 0.0


def test_paradigm_shift_crisis_detected():
    """Crisis when failure rate exceeds threshold."""
    engine = ParadigmShiftEngine(failure_threshold=0.3)
    for _ in range(7):
        engine.record_outcome("rust", False)
    for _ in range(3):
        engine.record_outcome("rust", True)

    is_crisis, rate = engine.check_crisis("rust")
    assert is_crisis
    assert rate > 0.3


def test_paradigm_shift_minimum_data():
    """No crisis with insufficient data."""
    engine = ParadigmShiftEngine(failure_threshold=0.3)
    engine.record_outcome("rust", False)

    is_crisis, rate = engine.check_crisis("rust")
    assert not is_crisis  # only 1 data point


def test_paradigm_shift_dialectic():
    """Test that run_dialectic returns a cognition task for the Agent."""
    engine = ParadigmShiftEngine(failure_threshold=0.3)
    result = engine.run_dialectic(
        None,  # No real engine needed
        "rust",
        "Always use ORM for database access",
        ["ORM caused 10x performance regression", "Raw SQL was 10x faster"],
    )

    assert result["status"] == "cognition_task"
    assert result["type"] == "dialectic_synthesis"
    assert result["domain"] == "rust"
    assert "thesis" in result
    assert "antithesis" in result
    assert "instruction" in result


def test_paradigm_shift_monitor():
    """Test monitor_and_shift with auto-trigger."""
    engine = ParadigmShiftEngine(failure_threshold=0.3)
    for _ in range(7):
        engine.record_outcome("db", False)
    for _ in range(3):
        engine.record_outcome("db", True)

    result = engine.monitor_and_shift(
        None,
        "db",
        "Always use ORM",
        ["ORM is slow"],
    )

    assert result is not None
    assert result["status"] == "cognition_task"

    # Counters should be reset
    counts = engine.failure_counts["db"]
    assert counts["successes"] == 0
    assert counts["failures"] == 0


def test_observer_no_intervention():
    """No intervention when system is healthy."""
    observer = ObserverNode(confidence_crisis_threshold=0.3, crisis_streak_length=5)

    # Simulate healthy queries
    for i in range(10):
        result = observer.observe(None, {"confidence": 0.8, "results": [{"layer": 1}, {"layer": 2}]})
        assert result is None


def test_observer_confidence_crisis():
    """Intervention when confidence is consistently low."""
    observer = ObserverNode(confidence_crisis_threshold=0.3, crisis_streak_length=5)

    # First 4 observations: no intervention (building history)
    for i in range(4):
        result = observer.observe(None, {"confidence": 0.1, "results": []})
        assert result is None

    # 5th observation triggers intervention (5 consecutive low-confidence)
    action = observer.observe(None, {"confidence": 0.1, "results": []})
    assert action is not None
    assert action["action"] == "signal_pushed"
    assert action["reason"] == "confidence_crisis"


def test_observer_layer_starvation():
    """Intervention when a layer has no results."""
    observer = ObserverNode()

    # Simulate 9 queries with only L2 results (L1 starving)
    for i in range(9):
        observer.observe(None, {"confidence": 0.8, "results": [{"layer": 2}]})

    # 10th observation triggers intervention (10 queries, L1 has 0 results)
    action = observer.observe(None, {"confidence": 0.8, "results": [{"layer": 2}]})
    assert action is not None
    assert action["action"] == "signal_pushed"


def test_observer_summary():
    """Test observation summary."""
    observer = ObserverNode()
    summary = observer.get_observation_summary()
    assert summary["status"] == "no_data"

    observer.observe(None, {"confidence": 0.5, "results": []})
    observer.observe(None, {"confidence": 0.7, "results": []})

    summary = observer.get_observation_summary()
    assert summary["queries_observed"] == 2
    assert summary["avg_confidence"] == 0.6


if __name__ == "__main__":
    tests = [
        test_paradigm_shift_no_crisis,
        test_paradigm_shift_crisis_detected,
        test_paradigm_shift_minimum_data,
        test_paradigm_shift_dialectic,
        test_paradigm_shift_monitor,
        test_observer_no_intervention,
        test_observer_confidence_crisis,
        test_observer_layer_starvation,
        test_observer_summary,
    ]
    for test in tests:
        test()
        print(f"  OK: {test.__name__}")
    print(f"\nAll {len(tests)} evolution tests passed!")
