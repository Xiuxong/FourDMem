"""Gap Reflection — Knowledge gap detection on cold start.

When the agent restarts after a period of inactivity, this module
compares the last known state with the current state and generates
a "gap reflection" report highlighting what changed.

Implements T-9.8: Knowledge gap reflection.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any


SNAPSHOT_DIR = "data/snapshots"


def save_snapshot(project_root: str, engine: Any) -> dict:
    """Save a snapshot of the current memory state.

    Called on session end or periodically.
    """
    snapshot_dir = os.path.join(project_root, SNAPSHOT_DIR)
    os.makedirs(snapshot_dir, exist_ok=True)

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stats": {},
        "recent_facts": [],
    }

    try:
        raw = engine.wake_up()
        stats = json.loads(raw) if isinstance(raw, str) else raw
        snapshot["stats"] = stats.get("memory_stats", stats)
    except Exception:
        pass

    try:
        tick = engine.get_tick()
        snapshot["tick"] = tick
    except Exception:
        pass

    # Save to file
    filepath = os.path.join(snapshot_dir, "last_session.json")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return snapshot


def load_last_snapshot(project_root: str) -> dict | None:
    """Load the last session snapshot."""
    filepath = os.path.join(project_root, SNAPSHOT_DIR, "last_session.json")
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def compute_gap(project_root: str, engine: Any) -> dict | None:
    """Compute the gap between last snapshot and current state.

    Returns:
        Gap report dict if a previous snapshot exists, None otherwise.
    """
    last = load_last_snapshot(project_root)
    if last is None:
        return None

    # Get current state
    current = {"stats": {}, "tick": 0}
    try:
        raw = engine.wake_up()
        stats = json.loads(raw) if isinstance(raw, str) else raw
        current["stats"] = stats.get("memory_stats", stats)
    except Exception:
        pass

    try:
        current["tick"] = engine.get_tick()
    except Exception:
        pass

    # Compute differences
    last_stats = last.get("stats", {})
    curr_stats = current.get("stats", {})

    l0_delta = curr_stats.get("l0_evidence", curr_stats.get("l0_count", 0)) - last_stats.get("l0_evidence", last_stats.get("l0_count", 0))
    l1_delta = curr_stats.get("l1_nodes", curr_stats.get("l1_node_count", 0)) - last_stats.get("l1_nodes", last_stats.get("l1_node_count", 0))
    tick_delta = current.get("tick", 0) - last.get("tick", 0)

    # Determine dormancy status
    last_ts = last.get("timestamp", "")
    dormancy_hours = 0
    if last_ts:
        try:
            last_dt = datetime.fromisoformat(last_ts)
            delta = datetime.now(timezone.utc) - last_dt
            dormancy_hours = delta.total_seconds() / 3600
        except Exception:
            pass

    needs_verification = dormancy_hours > 168  # > 7 days

    gap = {
        "last_session": last_ts,
        "dormancy_hours": round(dormancy_hours, 1),
        "tick_delta": tick_delta,
        "changes": {
            "l0_evidence_delta": l0_delta,
            "l1_nodes_delta": l1_delta,
        },
        "needs_verification": needs_verification,
        "recommendation": _generate_recommendation(
            dormancy_hours, l0_delta, l1_delta, needs_verification
        ),
    }

    return gap


def _generate_recommendation(
    dormancy_hours: float,
    l0_delta: int,
    l1_delta: int,
    needs_verification: bool,
) -> str:
    """Generate a human-readable recommendation based on the gap."""
    if dormancy_hours < 1:
        return "No significant gap. Continue normally."

    parts = []
    parts.append(f"Session resumed after {dormancy_hours:.0f} hours of inactivity.")

    if l0_delta > 0:
        parts.append(f"{l0_delta} new evidence items were added since last session.")
    if l1_delta > 0:
        parts.append(f"{l1_delta} new L1 facts were extracted.")
    if l1_delta < 0:
        parts.append(f"{abs(l1_delta)} L1 facts were pruned or removed.")

    if needs_verification:
        parts.append(
            "WARNING: Long dormancy (>7 days). Some memories may be outdated. "
            "Consider verifying critical facts before relying on them."
        )

    return " ".join(parts)
