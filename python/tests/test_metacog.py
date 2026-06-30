"""T-12.4: Metacognitive automatic drill-down accuracy tests."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fourdmem


def test_high_confidence_no_drill():
    """High confidence should not trigger drill-down."""
    engine = fourdmem.FourDMemEngine(":memory:")
    import json
    result = json.loads(engine.reflect("test topic", results_count=10, top_score=0.9))
    assert not result["should_drill_down"]


def test_low_confidence_triggers_drill():
    """Low confidence should trigger drill-down."""
    engine = fourdmem.FourDMemEngine(":memory:")
    import json
    result = json.loads(engine.reflect("obscure topic", results_count=0, top_score=0.0))
    assert result["should_drill_down"]


def test_thin_results_reduce_confidence():
    """Fewer results should lower confidence."""
    engine = fourdmem.FourDMemEngine(":memory:")
    import json
    thin = json.loads(engine.reflect("topic", results_count=1, top_score=0.9))
    thick = json.loads(engine.reflect("topic", results_count=10, top_score=0.9))
    assert thin["confidence"] < thick["confidence"]


if __name__ == "__main__":
    test_high_confidence_no_drill()
    test_low_confidence_triggers_drill()
    test_thin_results_reduce_confidence()
    print("All metacognition tests passed!")
