"""Evolution Scheduler — Background cognitive evolution daemon.

Runs as a daemon thread in the MCP Server. Periodically:
- Checks for pending cognitive signals (macro_candidate, paradigm_crisis, etc.)
- Triggers dream pruning every 100 ticks
- Triggers L1→L2 aggregation check
- Persists cognitive state on model changes

Architecture:
    MCP Server start → _start_evolution_scheduler() called
    Background thread loops, sleeping between checks
    All work is signal-driven — monitors push, scheduler checks
"""

import json
import os
import sys
import threading
import time
from typing import Any


try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

DREAM_CHECK_INTERVAL_SECONDS: int = 30
L1_L2_AGGREGATION_TICKS: int = 10
COGNITIVE_STATE_PERSIST_TICKS: int = 50
SIGNAL_CHECK_INTERVAL_SECONDS: int = 15


class EvolutionScheduler:
    """Background scheduler for cognitive evolution tasks.

    Spawns as a daemon thread. Wakes periodically to:
    1. Check dream pruning conditions (every 100 ticks)
    2. Persist cognitive state (every 50 ticks)
    3. Verify signal bus persistence
    """

    def __init__(self, project_root: str):
        self._project_root = project_root
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_dream_tick: int = 0
        self._last_persist_tick: int = 0

    def start(self):
        """Start the scheduler background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="evolution-scheduler")
        self._thread.start()
        logger.info("EvolutionScheduler started")

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                time.sleep(DREAM_CHECK_INTERVAL_SECONDS)
                self._tick()
            except Exception as e:
                logger.warning(f"EvolutionScheduler tick failed: {e}")

    def _tick(self):
        """Run one scheduler cycle."""
        engine = self._get_engine()
        if engine is None:
            return

        tick = self._get_tick(engine)

        # ── Dream pruning: every 100 ticks ──────────────────────────────
        if tick > 0 and tick - self._last_dream_tick >= 100:
            self._run_dream_pruning(engine)
            self._last_dream_tick = tick

        # ── Cognitive state persistence: every 50 ticks ─────────────────
        if tick - self._last_persist_tick >= 50:
            self._persist_cognitive_state(engine)
            self._last_persist_tick = tick

        # ── Signal persistence: every cycle ─────────────────────────────
        self._persist_signals()

    def _run_dream_pruning(self, engine: Any):
        """Run dream pruning if conditions met."""
        try:
            from mcp_server.lifecycle import _get_dream_pruner
            pruner = _get_dream_pruner()
            if pruner is not None:
                report = pruner.run_cycle(engine)
                if report.get("pruned", 0) > 0:
                    logger.info(f"Dream pruning: {report['pruned']} items pruned")
        except Exception as e:
            logger.debug(f"Dream pruning skipped: {e}")

    def _persist_cognitive_state(self, engine: Any):
        """Persist cognitive state + L1/L2 checkpoint to disk."""
        try:
            from mcp_server.state import _get_observer
            observer = _get_observer()
            if observer is not None:
                observer.persist_cognitive_state(engine, self._project_root)
        except Exception as e:
            logger.debug(f"Cognitive state persistence skipped: {e}")

        # Auto-checkpoint L1 facts/edges/versions to JSONL
        try:
            if engine.graph_node_count() > 0:
                vault_root = os.path.join(self._project_root, "data", "vault")
                engine.checkpoint(vault_root)
                logger.debug("L1 auto-checkpoint complete")
        except Exception as e:
            logger.debug(f"L1 auto-checkpoint skipped: {e}")

    def _persist_signals(self):
        """Persist signal bus to disk."""
        try:
            from cognition.signals import get_signal_bus
            bus = get_signal_bus()
            signals_path = os.path.join(self._project_root, "data", "cognition", "signals.json")
            bus.save(signals_path)
        except Exception:
            pass

    @staticmethod
    def _get_engine() -> Any:
        try:
            from mcp_server.state import get_engine
            return get_engine()
        except Exception:
            return None

    @staticmethod
    def _get_tick(engine: Any) -> int:
        try:
            return engine.get_tick()
        except Exception:
            return 0


# ── Module-level start ─────────────────────────────────────────────────────────

_scheduler: EvolutionScheduler | None = None


def start_scheduler(project_root: str):
    """Start the global evolution scheduler. Idempotent."""
    global _scheduler
    if _scheduler is not None:
        return
    _scheduler = EvolutionScheduler(project_root)
    _scheduler.start()


def stop_scheduler():
    """Stop the global evolution scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.stop()
        _scheduler = None
