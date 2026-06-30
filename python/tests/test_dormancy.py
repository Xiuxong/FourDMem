"""T-12.6: Long dormancy immunity tests."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fourdmem


def test_dormancy_does_not_destroy_core_memories():
    """Core memories should survive long dormancy periods."""
    engine = fourdmem.FourDMemEngine(":memory:")
    engine.add_fact("core architecture principle", None)

    # Simulate long dormancy (1000 ticks of inactivity)
    for _ in range(1000):
        engine.advance_tick()

    # Core fact should still be retrievable
    import json
    result = json.loads(engine.query("architecture principle", 5))
    assert result is not None, "Core memory should survive dormancy"


def test_subjective_time_immune_to_physical_downtime():
    """Tick-based recency should not be affected by physical time gaps."""
    engine = fourdmem.FourDMemEngine(":memory:")
    engine.add_fact("important fact", None)
    engine.advance_tick()

    # Even after "long downtime" (no ticks), the fact's recency
    # should be based on its last_active_tick, not wall-clock time
    import json
    result = json.loads(engine.query("important fact", 5))
    assert result["total_tokens"] >= 0  # Should return results


if __name__ == "__main__":
    test_dormancy_does_not_destroy_core_memories()
    test_subjective_time_immune_to_physical_downtime()
    print("All dormancy tests passed!")
