"""Cognitive Signal Bus — unified signal queue for Agent-driven cognition.

FourDMem monitors memory state (myelination, paradigm shift, topology,
dream pruning) and pushes *signals* to this bus. The Agent polls the bus
via `check_cognition_signals()` and decides which signals to act on.

Design principles:
- FourDMem *detects* patterns; Agent *decides* actions.
- Signals are advisory, never auto-modifying.
- Rate-limited to prevent flood; same-type signals coalesce.
- Thread-safe for concurrent producers (monitors run in background threads).

Signal types:
- macro_candidate: high-frequency successful query pattern detected
- paradigm_crisis: domain failure rate exceeded threshold
- phase_transition: graph topology reached critical complexity
- dream_pruning: decay candidates ready for review
- extraction_suggested: Agent hasn't called extract_deep recently
- observer_alert: retrieval system showing signs of inefficiency
"""

import threading
import time
from dataclasses import dataclass, field
import os
from typing import Any, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    """A cognitive signal pushed to the bus."""
    id: str
    type: str                # "macro_candidate" | "paradigm_crisis" | "phase_transition" | ...
    priority: int            # 0 (low) — 10 (critical)
    payload: dict[str, Any]  # context data for the Agent
    created_at: float        # time.monotonic()


# ── Signal bus ─────────────────────────────────────────────────────────────────

class SignalBus:
    """Thread-safe cognitive signal queue.

    Usage:
        bus = get_signal_bus()
        bus.push("macro_candidate", priority=3, payload={"pattern": "...", "stats": {...}})
        signals = bus.poll()  # Agent reads pending signals
        bus.ack("signal-id")  # Agent marks handled
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pending: list[Signal] = []
        self._counter: int = 0
        # Cooldown: signal_type -> last_push_time (monotonic seconds)
        self._cooldowns: dict[str, float] = {}
        # Cooldown durations in seconds per signal type
        self._cooldown_durations: dict[str, float] = {
            "macro_candidate": 300.0,      # 5 min
            "paradigm_crisis": 120.0,      # 2 min
            "phase_transition": 600.0,     # 10 min
            "dream_pruning": 3600.0,       # 1 hour
            "extraction_suggested": 120.0, # 2 min
            "observer_alert": 180.0,       # 3 min
            "re_evaluation": 600.0,        # 10 min — 条件变化重新评估
        }
        # Max pending signals (FIFO eviction)
        self._max_pending: int = 32

    def push(
        self,
        signal_type: str,
        priority: int = 5,
        payload: Optional[dict[str, Any]] = None,
        coalesce_key: Optional[str] = None,
    ) -> Optional[str]:
        """Push a signal to the bus. Returns signal ID if pushed, None if rate-limited.

        Args:
            signal_type: One of the recognized signal types.
            priority: 0 (background) to 10 (critical).
            payload: Arbitrary context dict for the Agent.
            coalesce_key: If set, replaces any pending signal of same type with
                          matching coalesce_key (dedup).

        Returns:
            Signal ID if pushed, None if suppressed by cooldown.
        """
        now = time.monotonic()
        cooldown = self._cooldown_durations.get(signal_type, 60.0)

        with self._lock:
            # Rate-limit check
            last = self._cooldowns.get(signal_type, 0.0)
            if now - last < cooldown:
                return None
            self._cooldowns[signal_type] = now

            # Coalesce: replace existing signal of same type+key
            if coalesce_key is not None:
                for i, s in enumerate(self._pending):
                    if s.type == signal_type and s.payload.get("coalesce_key") == coalesce_key:
                        s.payload = (payload or {}) | {"coalesce_key": coalesce_key}
                        s.priority = priority
                        s.created_at = now
                        return s.id

            # Evict oldest if at capacity
            while len(self._pending) >= self._max_pending:
                self._pending.pop(0)

            # Create and enqueue
            self._counter += 1
            sig_id = f"sig:{signal_type}:{self._counter}"
            full_payload = (payload or {}) | ({"coalesce_key": coalesce_key} if coalesce_key else {})
            signal = Signal(
                id=sig_id,
                type=signal_type,
                priority=priority,
                payload=full_payload,
                created_at=now,
            )
            self._pending.append(signal)
            logger.debug(f"Signal pushed: {sig_id} (type={signal_type}, pri={priority})")
            return sig_id

    def poll(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return pending signals ordered by priority desc, then FIFO.

        Does NOT remove signals — call ack() to mark handled.
        """
        with self._lock:
            sorted_signals = sorted(
                self._pending, key=lambda s: (-s.priority, s.created_at)
            )
            result = sorted_signals[:limit]
            return [
                {
                    "id": s.id,
                    "type": s.type,
                    "priority": s.priority,
                    "payload": s.payload,
                }
                for s in result
            ]

    def ack(self, signal_id: str) -> bool:
        """Mark a signal as handled (remove from queue)."""
        with self._lock:
            for i, s in enumerate(self._pending):
                if s.id == signal_id:
                    self._pending.pop(i)
                    return True
            return False

    def ack_type(self, signal_type: str) -> int:
        """Acknowledge all signals of a given type. Returns count removed."""
        with self._lock:
            before = len(self._pending)
            self._pending = [s for s in self._pending if s.type != signal_type]
            return before - len(self._pending)

    def pending_count(self) -> int:
        """Number of unhandled signals."""
        with self._lock:
            return len(self._pending)

    def clear(self):
        """Remove all pending signals (e.g. on session reset)."""
        with self._lock:
            self._pending.clear()
            self._cooldowns.clear()

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> int:
        """Persist pending signals to a JSON file. Returns count saved."""
        import json as _json
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with self._lock:
                data = {
                    "signals": [
                        {
                            "id": s.id,
                            "type": s.type,
                            "priority": s.priority,
                            "payload": s.payload,
                            "created_at": s.created_at,
                        }
                        for s in self._pending
                    ],
                    "counter": self._counter,
                }
            with open(path, "w", encoding="utf-8") as f:
                _json.dump(data, f, indent=2, ensure_ascii=False)
            return len(data["signals"])
        except OSError:
            return 0

    def load(self, path: str) -> int:
        """Load persisted signals from a JSON file. Returns count loaded."""
        import json as _json
        if not os.path.exists(path):
            return 0
        try:
            with open(path, encoding="utf-8") as f:
                data = _json.load(f)
            with self._lock:
                for s_data in data.get("signals", []):
                    if len(self._pending) >= self._max_pending:
                        break
                    if any(s.id == s_data["id"] for s in self._pending):
                        continue
                    self._pending.append(Signal(
                        id=s_data["id"],
                        type=s_data["type"],
                        priority=s_data["priority"],
                        payload=s_data.get("payload", {}),
                        created_at=s_data.get("created_at", time.monotonic()),
                    ))
                self._counter = max(self._counter, data.get("counter", 0))
            return len(data.get("signals", []))
        except (OSError, _json.JSONDecodeError, KeyError):
            return 0


# ── Singleton access ───────────────────────────────────────────────────────────

_signal_bus: Optional[SignalBus] = None
_signal_bus_lock = threading.Lock()


def get_signal_bus() -> SignalBus:
    """Get or create the global SignalBus singleton."""
    global _signal_bus
    if _signal_bus is None:
        with _signal_bus_lock:
            if _signal_bus is None:
                _signal_bus = SignalBus()
    return _signal_bus


def reset_signal_bus():
    """Reset the signal bus (for testing)."""
    global _signal_bus
    with _signal_bus_lock:
        _signal_bus = None
