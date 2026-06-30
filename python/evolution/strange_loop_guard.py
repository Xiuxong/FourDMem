"""Strange Loop Safety Guard — Proposal→Verify→Commit governance for self-modification.

Implements the safety gate for ObserverNode's parameter modifications:
- Propose: validate against immutable constraints, delta limits, cooldown
- Verify: check transferability (OEP defense — single-evidence penalty)
- Commit: apply with snapshot for rollback
- Rollback: restore previous state from snapshot
"""

from dataclasses import dataclass, field
from typing import Any, Optional


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class ModificationProposal:
    """A proposed parameter modification from the Observer."""
    id: str = ""
    target: str = ""
    current_value: Any = None
    proposed_value: Any = None
    justification: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    tick: int = 0


@dataclass
class VerificationResult:
    """Result of transferability verification."""
    passed: bool = False
    score: float = 0.0
    generalization_score: float = 0.0
    reason: str = ""


# ── Constants ──────────────────────────────────────────────────────────────────

MAX_WEIGHT_DELTA = 0.1
MIN_RIF_WEIGHT = 0.05
COOLDOWN_TICKS = 5
MAX_PROPOSALS_PER_TICK = 2

FORBIDDEN_TARGETS = {
    "delete_evidence",
    "drop_index",
    "truncate_table",
    "purge_l0",
    "modify_schema",
    "disable_governor",
}

CONSTITUTIONAL_RULES = {
    "never_delete_raw_evidence",
    "always_preserve_provenance_chain",
    "maintain_version_history",
    "no_user_data_exfiltration",
}


# ── StrangeLoopGuard ──────────────────────────────────────────────────────────

class StrangeLoopGuard:
    """Safety guard for Observer self-modification proposals.

    Enforces:
    - Immutable target protection
    - Delta magnitude limits
    - Cooldown between interventions
    - Rate limiting per tick
    - Transferability verification (OEP defense)
    - Snapshot-based rollback
    """

    def __init__(self):
        self._last_proposal_tick: int = -COOLDOWN_TICKS
        self._proposals_this_tick: int = 0
        self._current_tick: int = -1
        self._committed: int = 0
        self._snapshots: dict[str, dict] = {}  # proposal_id → snapshot

    def propose(
        self,
        target: str,
        current_value: Any = None,
        proposed_value: Any = None,
        justification: str = "",
        evidence_ids: list[str] | None = None,
        tick: int = 0,
    ) -> tuple[bool, str, ModificationProposal | None]:
        """Validate a modification proposal against safety constraints.

        Returns (accepted, message, proposal_or_None).
        """
        # Reset per-tick counter
        if tick != self._current_tick:
            self._current_tick = tick
            self._proposals_this_tick = 0

        # 1. Immutable target check
        if target in FORBIDDEN_TARGETS:
            return False, "Target is immutable — forbidden by safety constraints", None

        # 2. Constitutional rule violation check
        for rule in CONSTITUTIONAL_RULES:
            if rule.replace("_", " ") in target.lower() or target.lower() in rule:
                return False, "Constitutional rule violation — cannot modify", None

        # 3. Numeric constraint checks (for numeric values)
        if isinstance(current_value, (int, float)) and isinstance(proposed_value, (int, float)):
            # Floor check first — below-floor always rejected regardless of delta
            if proposed_value < MIN_RIF_WEIGHT and "rif_weights" in target:
                return False, f"Proposed value {proposed_value} below floor {MIN_RIF_WEIGHT}", None

            # Delta magnitude check
            delta = abs(proposed_value - current_value)
            if delta > MAX_WEIGHT_DELTA:
                return False, f"Delta {delta:.3f} exceeds max_weight_delta {MAX_WEIGHT_DELTA}", None

        # 4. Cooldown check (skip if same tick — rate limit handles that)
        if tick != self._last_proposal_tick and tick - self._last_proposal_tick < COOLDOWN_TICKS:
            return False, f"Cooldown active — {COOLDOWN_TICKS} ticks required between proposals", None

        # 5. Rate limit check
        if self._proposals_this_tick >= MAX_PROPOSALS_PER_TICK:
            return False, f"Rate limit exceeded — max {MAX_PROPOSALS_PER_TICK} proposals per tick", None

        # Accept proposal
        self._last_proposal_tick = tick
        self._proposals_this_tick += 1

        proposal = ModificationProposal(
            id=f"prop-{tick}-{self._proposals_this_tick}",
            target=target,
            current_value=current_value,
            proposed_value=proposed_value,
            justification=justification,
            evidence_ids=evidence_ids or [],
            tick=tick,
        )
        return True, "Proposal accepted", proposal

    def verify(
        self,
        proposal: ModificationProposal,
        test_cases: list[dict] | None = None,
    ) -> VerificationResult:
        """Verify transferability of a proposal (OEP defense).

        Single evidence source → low generalization (overfitting risk).
        Multiple evidence sources → high generalization.
        """
        n_evidence = len(proposal.evidence_ids) if proposal.evidence_ids else 0

        if test_cases is None and n_evidence == 0:
            # No verification data — basic check passes
            return VerificationResult(passed=True, score=1.0, generalization_score=1.0)

        # OEP defense: generalization scales with evidence diversity
        if n_evidence == 0:
            generalization = 1.0
        elif n_evidence == 1:
            generalization = 0.4  # Low — single source, overfitting risk
        elif n_evidence <= 3:
            generalization = 0.7  # Moderate
        else:
            generalization = min(0.9 + (n_evidence - 4) * 0.02, 1.0)  # High

        passed = generalization >= 0.3
        return VerificationResult(
            passed=passed,
            score=generalization,
            generalization_score=generalization,
        )

    def commit(
        self,
        proposal: ModificationProposal,
        verification: VerificationResult,
    ) -> tuple[bool, str]:
        """Commit a verified proposal, saving a snapshot for rollback."""
        if not verification.passed:
            return False, "Commit blocked — verification not passed"

        # Save snapshot for rollback
        self._snapshots[proposal.id] = {
            "target": proposal.target,
            "previous_value": proposal.current_value,
            "new_value": proposal.proposed_value,
        }
        self._committed += 1
        return True, f"Committed proposal {proposal.id}"

    def rollback(self, proposal_id: str) -> tuple[bool, str]:
        """Rollback a committed proposal using its snapshot."""
        if proposal_id not in self._snapshots:
            return False, f"No snapshot found for proposal {proposal_id}"

        snapshot = self._snapshots.pop(proposal_id)
        return True, f"Rolled back {snapshot['target']} to {snapshot['previous_value']}"

    def get_stats(self) -> dict:
        """Return governance statistics."""
        return {
            "committed": self._committed,
            "active_snapshots": len(self._snapshots),
            "last_proposal_tick": self._last_proposal_tick,
        }
