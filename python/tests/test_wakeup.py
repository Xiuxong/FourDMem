"""T-12.7: Cold-start wake-up and environment alignment tests."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fourdmem


def test_wake_up_returns_state():
    """Wake-up should return current memory state summary."""
    engine = fourdmem.FourDMemEngine(":memory:")
    engine.add_fact("test fact", None)
    engine.advance_tick()

    import json
    result = json.loads(engine.wake_up())
    assert result["status"] == "awake"
    assert "memory_stats" in result
    assert "current_tick" in result


def test_wake_up_empty_engine():
    """Wake-up on empty engine should return valid state."""
    engine = fourdmem.FourDMemEngine(":memory:")
    import json
    result = json.loads(engine.wake_up())
    assert result["status"] == "awake"
    assert result["memory_stats"]["l0_evidence"] == 0


def test_wake_up_after_ingestion():
    """Wake-up should reflect ingested evidence."""
    engine = fourdmem.FourDMemEngine(":memory:")
    engine.save("session-1", "user", "hello world", "{}")
    engine.save("session-1", "assistant", "hi there", "{}")

    import json
    result = json.loads(engine.wake_up())
    assert result["memory_stats"]["l0_evidence"] >= 2


if __name__ == "__main__":
    test_wake_up_returns_state()
    test_wake_up_empty_engine()
    test_wake_up_after_ingestion()
    print("All wake-up tests passed!")
