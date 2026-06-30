"""T-12.3: Dream pruning and signal-to-noise ratio tests."""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fourdmem


def test_dream_prune_removes_decayed():
    """Dream pruning should remove facts beyond their shelf life."""
    engine = fourdmem.FourDMemEngine(":memory:")
    engine.add_fact("ephemeral context", None)

    # Advance far beyond default shelf life (90 ticks)
    for _ in range(200):
        engine.advance_tick()

    import json
    report = json.loads(engine.dream_prune(100, 0.7))
    assert report["pruned"] >= 1, "Should prune at least 1 decayed fact"


def test_dream_prune_preserves_high_utility():
    """High-utility facts should be immune from pruning."""
    engine = fourdmem.FourDMemEngine(":memory:")
    idx = engine.add_fact("critical architecture decision", None)
    engine.feedback("architecture decision", 0.9)

    for _ in range(200):
        engine.advance_tick()

    import json
    report = json.loads(engine.dream_prune(100, 0.7))
    assert report["preserved"] >= 1, "Should preserve high-utility fact"


def test_dream_preserves_pain_points():
    """Pain-point marked facts should never be pruned."""
    engine = fourdmem.FourDMemEngine(":memory:")
    r = json.loads(engine.add_fact("critical database migration bug", None))
    engine.mark_pain_point(r["node_index"])

    for _ in range(500):
        engine.advance_tick()

    report = json.loads(engine.dream_prune(100, 0.7))
    assert report["preserved"] >= 1, "Pain-point should be preserved"


if __name__ == "__main__":
    test_dream_prune_removes_decayed()
    test_dream_prune_preserves_high_utility()
    test_dream_preserves_pain_points()
    print("All dream pruning tests passed!")
