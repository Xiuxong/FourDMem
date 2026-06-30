"""P0 Integration Tests — SSGM + L2 + StrangeLoop governance gates.

Validates:
1. SSGM: governance gate prevents unsafe writes
2. L2: dual channel index search works
3. StrangeLoop: proposal→verify→commit flow with rollback
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from cognition.ssgm_governor import (
    SSGMGovernor, AccessRequest, GovernanceDecision,
    IMMUTABLE_CONSTRAINTS,
)
from cognition.l2_indexer import (
    L2DualChannelIndex, NarrativeSchema,
)
from cognition.ssgm_governor import IMMUTABLE_CONSTRAINTS
STRANGE_LOOP_IMMUTABLES = IMMUTABLE_CONSTRAINTS  # alias for test compatibility
from evolution.strange_loop_guard import StrangeLoopGuard, ModificationProposal, VerificationResult


# ═══════════════════════════════════════════════════════════════════
# SSGM Governance Tests
# ═══════════════════════════════════════════════════════════════════

class TestSSGMGovernance:
    """SSGM governance gate tests."""

    def test_pre_consolidation_no_conflict(self):
        """New fact with no existing contradictions should be allowed."""
        gov = SSGMGovernor()
        existing = [
            {"id": "fact-1", "content": "Rust uses async/await", "embedding": [0.1] * 768},
        ]
        decision = gov.pre_consolidation_check(
            "Python uses asyncio", existing, current_tick=10,
        )
        assert decision.action == "allow"

    def test_access_control_shared(self):
        """Shared visibility should grant access to all agents."""
        gov = SSGMGovernor()
        req = AccessRequest(
            memory_id="mem-1",
            workspace_id="project-a",
            agent_id="agent-1",
            memory_visibility="shared",
            memory_workspace_id="project-b",
        )
        decision = gov.access_control(req)
        assert decision.access_granted is True

    def test_access_control_workspace_isolation(self):
        """Different workspace agents should be denied workspace-scoped memory."""
        gov = SSGMGovernor()
        req = AccessRequest(
            memory_id="mem-1",
            workspace_id="project-a",
            agent_id="agent-1",
            memory_visibility="workspace",
            memory_workspace_id="project-b",  # different workspace
        )
        decision = gov.access_control(req)
        assert decision.access_granted is False

    def test_access_control_same_workspace_allowed(self):
        """Same workspace agents should be allowed workspace-scoped memory."""
        gov = SSGMGovernor()
        req = AccessRequest(
            memory_id="mem-1",
            workspace_id="project-a",
            agent_id="agent-1",
            memory_visibility="workspace",
            memory_workspace_id="project-a",  # same workspace
        )
        decision = gov.access_control(req)
        assert decision.access_granted is True

    def test_temporal_decay_immune(self):
        """Immune shelf-life should never decay."""
        gov = SSGMGovernor()
        memory = {
            "last_active_tick": 0,
            "utility_score": 0.3,
            "half_life_ticks": 30,
            "shelf_life": "immune",
        }
        retention = gov.temporal_decay_model(memory, current_tick=1000)
        assert retention == 1.0

    def test_temporal_decay_active(self):
        """Active memory should decay with ticks."""
        gov = SSGMGovernor()
        memory = {
            "last_active_tick": 0,
            "utility_score": 0.3,
            "half_life_ticks": 50,
            "shelf_life": "subjective",
        }
        retention_early = gov.temporal_decay_model(memory, current_tick=10)
        retention_late = gov.temporal_decay_model(memory, current_tick=200)
        assert retention_early > retention_late
        assert retention_early > 0.8   # should be high early
        assert retention_late < 0.5    # should decay significantly

    def test_high_utility_immunity(self):
        """High utility (>0.8) should resist decay."""
        gov = SSGMGovernor()
        memory = {
            "last_active_tick": 0,
            "utility_score": 0.9,
            "half_life_ticks": 30,
            "shelf_life": "subjective",
        }
        retention = gov.temporal_decay_model(memory, current_tick=100)
        assert retention >= 0.9  # high utility → decay immunity

    def test_immutable_target_rejected(self):
        """Modifying a forbidden target must be rejected."""
        gov = SSGMGovernor()
        decision = gov.pre_evolution_check(
            target="delete_evidence",
            proposed_value=True,
            current_value=False,
            current_tick=10,
        )
        assert decision.action == "reject"
        assert "immutable" in decision.reason.lower()


# ═══════════════════════════════════════════════════════════════════
# L2 Dual Channel Tests
# ═══════════════════════════════════════════════════════════════════

class TestL2DualChannel:
    """L2 dual channel index tests."""

    def test_schema_creation(self):
        """NarrativeSchema should serialize correctly."""
        schema = NarrativeSchema(
            trigger_event="tokio panic in production",
            context="M11 project, Rust async runtime",
            action="Fixed by adding timeout to channel send",
            outcome="Panic resolved, added test coverage",
            lesson="Always add timeouts to unbounded channels",
            workspace_id="m11",
        )
        d = schema.to_dict()
        assert d["trigger_event"] == "tokio panic in production"
        assert d["workspace_id"] == "m11"
        assert len(d["id"]) == 12

    def test_schema_text_representation(self):
        """Schema text should be searchable."""
        schema = NarrativeSchema(
            trigger_event="bug: memory leak",
            context="FourDMem project",
            action="Fixed Arc cycle",
            outcome="Memory stabilized",
            lesson="Use Weak<T> for parent references",
        )
        text = schema.to_text()
        assert "memory leak" in text
        assert "FourDMem" in text
        assert "Weak<T>" in text

    def test_l2_index_creation(self):
        """L2DualChannelIndex should initialize with empty state."""
        l2 = L2DualChannelIndex()
        stats = l2.get_stats()
        assert stats["embedding_dim"] == 768
        assert stats["episodic_schemata"] == 0

    def test_episodic_search_empty(self):
        """Empty index should return empty results."""
        l2 = L2DualChannelIndex()
        results = l2.search_episodic("how to fix memory leak", top_k=5)
        assert results == []

    def test_episodic_search_with_schema(self):
        """Adding schemata should make them searchable."""
        l2 = L2DualChannelIndex()
        schema = NarrativeSchema(
            trigger_event="Rust async runtime crash",
            context="tokio project",
            action="debugged with tracing",
            outcome="fixed",
            lesson="use spawn_blocking for CPU-heavy tasks",
        )
        l2.add_schema(schema)
        results = l2.search_episodic("async runtime crash", top_k=5)
        assert len(results) >= 1
        assert results[0]["channel"] == "episodic"

    def test_tracemem_extraction(self):
        """TraceMem extraction should create schemata from conversations."""
        l2 = L2DualChannelIndex()
        schema = l2.extract_schema_from_turn(
            user_message="报错：Connection refused，如何修复？",
            assistant_response="方案：检查目标端口是否启动，修改 firewall 规则",
            workspace_id="test",
        )
        assert schema is not None
        assert "Connection refused" in schema.trigger_event
        assert schema.workspace_id == "test"

    def test_unified_search(self):
        """Unified search API should return both channels."""
        l2 = L2DualChannelIndex()
        schema = NarrativeSchema(
            trigger_event="python asyncio deadlock",
            lesson="avoid mixing sync and async code",
        )
        l2.add_schema(schema)
        result = l2.search("asyncio deadlock", top_k=5)
        assert "query" in result
        assert "results" in result
        assert "semantic_count" in result
        assert "episodic_count" in result


# ═══════════════════════════════════════════════════════════════════
# Strange Loop Guard Tests
# ═══════════════════════════════════════════════════════════════════

class TestStrangeLoopGuard:
    """Strange Loop safety guard tests."""

    def test_propose_valid(self):
        """Valid proposal within bounds should be accepted."""
        guard = StrangeLoopGuard()
        ok, msg, prop = guard.propose(
            target="rif_weights.recency",
            current_value=0.25,
            proposed_value=0.27,
            justification="Improve recency weight for recent queries",
            tick=10,
        )
        assert ok is True
        assert prop is not None
        assert prop.target == "rif_weights.recency"

    def test_propose_forbidden_target(self):
        """Forbidden target modification must be rejected."""
        guard = StrangeLoopGuard()
        ok, msg, prop = guard.propose(
            target="delete_evidence",
            current_value=False,
            proposed_value=True,
            justification="Clean old data",
            tick=10,
        )
        assert ok is False
        assert "immutable" in msg.lower()

    def test_propose_delta_too_large(self):
        """Delta exceeding max_weight_delta must be rejected."""
        guard = StrangeLoopGuard()
        ok, msg, prop = guard.propose(
            target="rif_weights.importance",
            current_value=0.35,
            proposed_value=0.80,  # delta > 0.1
            justification="Importance should be higher",
            tick=10,
        )
        assert ok is False
        assert "delta" in msg.lower()

    def test_propose_below_floor(self):
        """Value below min_rif_weight floor must be rejected."""
        guard = StrangeLoopGuard()
        ok, msg, prop = guard.propose(
            target="rif_weights.recency",
            current_value=0.25,
            proposed_value=0.01,  # below 0.05 floor
            justification="Reduce recency",
            tick=10,
        )
        assert ok is False
        assert "floor" in msg.lower()

    def test_propose_cooldown(self):
        """Proposal within cooldown period must be rejected."""
        guard = StrangeLoopGuard()
        # First proposal at tick 5
        guard.propose("rif_weights.recency", 0.25, 0.27, tick=5)
        # Second proposal at tick 8 (within 5-tick cooldown)
        ok, msg, _ = guard.propose(
            "rif_weights.frequency", 0.15, 0.17, tick=8,
        )
        assert ok is False
        assert "cooldown" in msg.lower()

    def test_verify_no_test_cases(self):
        """Verification without test cases should pass basic check."""
        guard = StrangeLoopGuard()
        prop = ModificationProposal(
            id="test-1",
            target="rif_weights.recency",
            current_value=0.25,
            proposed_value=0.27,
        )
        result = guard.verify(prop, test_cases=None)
        assert result.passed is True
        assert result.score == 1.0

    def test_verify_with_evidence(self):
        """Multiple evidence sources should improve transferability."""
        guard = StrangeLoopGuard()
        prop = ModificationProposal(
            id="test-2",
            target="rif_weights.recency",
            current_value=0.25,
            proposed_value=0.27,
            evidence_ids=["e1", "e2", "e3", "e4", "e5"],
        )
        result = guard.verify(prop, test_cases=None)
        assert result.generalization_score >= 0.9  # 5 evidence → high confidence

    def test_verify_single_evidence_low_transferability(self):
        """Single evidence source should have lower transferability (OEP defense)."""
        guard = StrangeLoopGuard()
        prop = ModificationProposal(
            id="test-3",
            target="rif_weights.recency",
            current_value=0.25,
            proposed_value=0.27,
            evidence_ids=["e1"],  # single source
        )
        result = guard.verify(prop, test_cases=None)
        assert result.generalization_score < 0.6  # OEP: low diversity → low score

    def test_commit_blocked_without_verification(self):
        """Commit should be blocked if verification not passed."""
        guard = StrangeLoopGuard()
        prop = ModificationProposal(id="t", target="x", current_value=1, proposed_value=2)
        bad_verify = VerificationResult(passed=False, score=0.0)
        ok, msg = guard.commit(prop, bad_verify)
        assert ok is False
        assert "blocked" in msg.lower()

    def test_commit_with_verification(self):
        """Verified proposal should be committed with snapshot."""
        guard = StrangeLoopGuard()
        prop = ModificationProposal(id="t", target="x", current_value=1, proposed_value=2)
        good_verify = VerificationResult(passed=True, score=1.0, generalization_score=0.9)
        ok, msg = guard.commit(prop, good_verify)
        assert ok is True
        stats = guard.get_stats()
        assert stats["committed"] == 1

    def test_rollback(self):
        """Rollback should restore snapshot."""
        guard = StrangeLoopGuard()
        prop = ModificationProposal(id="t", target="x", current_value=1, proposed_value=2)
        good_verify = VerificationResult(passed=True, score=1.0, generalization_score=0.9)
        guard.commit(prop, good_verify)
        ok, msg = guard.rollback("t")
        assert ok is True
        stats = guard.get_stats()
        assert stats["active_snapshots"] == 0

    def test_propose_rate_limit_per_tick(self):
        """Should reject proposals exceeding max_proposals_per_tick."""
        guard = StrangeLoopGuard()
        for i in range(3):  # max is 2 per tick
            # Use small deltas to avoid delta check interference
            ok, _, _ = guard.propose(
                f"rif_weights.test_{i}", 0.25, 0.26, tick=10,
            )
            if i < 2:
                assert ok is True, f"Proposal {i} should be accepted, got: {ok}"
            else:
                assert ok is False, f"Proposal {i} should be rejected"

    def test_constitutional_rule_violation(self):
        """Constitutional rule violation must be rejected."""
        guard = StrangeLoopGuard()
        ok, msg, _ = guard.propose(
            target="disable_governor",
            current_value=False,
            proposed_value=True,
            justification="Remove governance",
            tick=10,
        )
        assert ok is False
    def test_propose_with_justification_and_evidence(self):
        """Full proposal with justification and evidence should be accepted."""
        guard = StrangeLoopGuard()
        ok, msg, prop = guard.propose(
            target="rif_weights.utility",
            current_value=0.25,
            proposed_value=0.27,  # delta 0.08, within 0.1 limit
            justification="Utility feedback shows high correlation with task success",
            evidence_ids=["ev-1", "ev-2", "ev-3"],
            tick=20,  # different tick to avoid cooldown from previous tests
        )
        assert ok is True, f"Expected proposal to be accepted: {msg}"
        assert prop is not None
        assert prop.justification != ""
        assert len(prop.evidence_ids) == 3

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
