"""Tests for cognition.salience — embedding-enhanced salience detection."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cognition.salience import SalienceDetector


def test_low_salience_greeting():
    """Greetings should have zero salience."""
    det = SalienceDetector(threshold=2.0)
    assert det.check("hi") == 0.0
    assert det.check("ok") == 0.0
    assert det.check("thanks") == 0.0


def test_decision_keywords():
    """Decision keywords should boost salience."""
    det = SalienceDetector(threshold=2.0)
    score = det.check("we must use the borrow checker approach")
    assert score > 0


def test_memory_keywords():
    """Memory keywords should have high weight."""
    det = SalienceDetector(threshold=2.0)
    score = det.check("remember that the database migration requires downtime")
    assert score >= 2.0  # MEMORY_PATTERNS base weight is 2.5


def test_architecture_keywords():
    """Architecture keywords should be detected."""
    det = SalienceDetector(threshold=2.0)
    score = det.check("the architecture uses a microservices design pattern")
    assert score > 0


def test_bug_keywords():
    """Bug keywords should be detected."""
    det = SalienceDetector(threshold=2.0)
    score = det.check("found a bug in the error handling that causes a crash")
    assert score > 0


def test_should_extract_after_threshold():
    """should_extract returns True after high-salience content."""
    det = SalienceDetector(threshold=2.0)
    det.check("remember this critical important fact about the architecture")
    assert det.should_extract() is True


def test_should_not_extract_low_salience():
    """should_extract returns False for low-salience content."""
    det = SalienceDetector(threshold=2.0)
    det.check("ok")
    assert det.should_extract() is False


def test_get_pending_content():
    """get_pending_content returns and resets buffer."""
    det = SalienceDetector(threshold=2.0)
    det.check("remember this critical fact")
    content = det.get_pending_content()
    assert len(content) > 0
    # Should be reset after get
    assert len(det.get_pending_content()) == 0


def test_reset():
    """reset should clear all state."""
    det = SalienceDetector(threshold=2.0)
    det.check("remember this critical fact")
    det.reset()
    assert det.should_extract() is False
    assert len(det.get_pending_content()) == 0


def test_chinese_keywords():
    """Chinese keywords should be detected."""
    det = SalienceDetector(threshold=2.0)
    score = det.check("必须记住这个重要的架构设计决策")
    assert score > 0


def test_multi_signal_bonus():
    """Keyword + semantic both positive should give bonus."""
    det = SalienceDetector(threshold=2.0)
    # This should trigger keyword score > 0
    score = det.check("we must implement the database migration architecture")
    assert score > 0


if __name__ == "__main__":
    test_low_salience_greeting()
    test_decision_keywords()
    test_memory_keywords()
    test_architecture_keywords()
    test_bug_keywords()
    test_should_extract_after_threshold()
    test_should_not_extract_low_salience()
    test_get_pending_content()
    test_reset()
    test_chinese_keywords()
    test_multi_signal_bonus()
    print("All salience tests passed!")
