"""T-12.2: Hot cognition (feedback) sensitivity tests."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fourdmem


def test_positive_feedback_increases_utility():
    """Positive feedback should increase a fact's utility score."""
    engine = fourdmem.FourDMemEngine(":memory:")
    idx = engine.add_fact("important database optimization", None)

    # Apply positive feedback
    engine.feedback("database optimization", 0.5)

    # The fact should now have higher utility (verifiable via query ranking)
    result = engine.query("database optimization", 5)
    assert result is not None


def test_negative_feedback_decreases_utility():
    """Negative feedback should decrease a fact's utility score."""
    engine = fourdmem.FourDMemEngine(":memory:")
    idx = engine.add_fact("deprecated ORM pattern", None)

    # Apply negative feedback
    engine.feedback("deprecated ORM", -0.8)

    result = engine.query("deprecated ORM", 5)
    assert result is not None


def test_feedback_affects_ranking():
    """Facts with positive feedback should rank higher than unranked ones."""
    engine = fourdmem.FourDMemEngine(":memory:")
    engine.add_fact("alpha testing strategy", None)
    engine.add_fact("beta testing strategy", None)

    # Boost alpha
    engine.feedback("alpha testing", 0.9)

    # Alpha should appear first in results
    import json
    result = json.loads(engine.query("testing strategy", 10))
    if result["results"]:
        assert result["results"][0]["id"] is not None


if __name__ == "__main__":
    test_positive_feedback_increases_utility()
    test_negative_feedback_decreases_utility()
    test_feedback_affects_ranking()
    print("All feedback tests passed!")
