"""T-12.1: Cognitive dissonance and conflict resolution tests."""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fourdmem


def test_conflict_detection():
    """Adding two contradictory facts should create conflict edges."""
    engine = fourdmem.FourDMemEngine(":memory:")

    r1 = json.loads(engine.add_fact("rust borrow checker prevents data races", None))
    r2 = json.loads(engine.add_fact("rust borrow checker ensures memory safety", None))
    idx1 = r1["node_index"]

    conflicts = engine.resolve_conflicts(idx1)
    assert conflicts >= 1, "Should detect at least 1 conflict between similar facts"


def test_no_conflict_for_different_facts():
    """Completely unrelated facts should not trigger conflicts."""
    engine = fourdmem.FourDMemEngine(":memory:")

    r1 = json.loads(engine.add_fact("database migration strategy", None))
    r2 = json.loads(engine.add_fact("color theory for UI design", None))
    idx1 = r1["node_index"]

    conflicts = engine.resolve_conflicts(idx1)
    assert conflicts == 0, "Unrelated facts should have no conflicts"


if __name__ == "__main__":
    test_conflict_detection()
    test_no_conflict_for_different_facts()
    print("All conflict tests passed!")
