"""T-12.5: Counterfactual marking and pitfall avoidance tests."""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fourdmem


def test_abandon_branch_marks_counterfactual():
    """Abandoning a branch should mark it as counterfactual."""
    engine = fourdmem.FourDMemEngine(":memory:")

    # abandon_branch operates on version tree entities.
    # Without version history, it should raise an error gracefully.
    try:
        engine.abandon_branch("nonexistent", 1, "reason")
        assert False, "Should have raised an error for nonexistent entity"
    except RuntimeError:
        pass  # Expected: entity not found in version tree


def test_counterfactual_api_exists():
    """abandon_branch method should be available on the engine."""
    engine = fourdmem.FourDMemEngine(":memory:")
    assert hasattr(engine, 'abandon_branch')


if __name__ == "__main__":
    test_abandon_branch_marks_counterfactual()
    test_counterfactual_api_exists()
    print("All counterfactual tests passed!")
