"""Health Monitor — MCP Server liveness probes and auto-restart.

Periodically sends wake_up probes to the FourDMem engine.
If consecutive probes fail, triggers an alert via the signal bus.
Designed for embedding in process supervisors (systemd, NSSM, supervisord).

Usage (standalone):
    python -m daemon.health_monitor --db-path PATH --interval 30 --max-failures 3

Usage (embedded in MCP Server):
    from daemon.health_monitor import HealthMonitor
    monitor = HealthMonitor(project_root, check_interval_s=60)
    monitor.start()
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


class HealthMonitor:
    """Monitors FourDMem engine health via periodic wake_up probes.

    On consecutive failures, pushes an observer_alert signal to the
    cognitive signal bus. Does NOT auto-restart — that's the process
    supervisor's responsibility.
    """

    def __init__(
        self,
        project_root: str,
        check_interval_s: int = 60,
        max_consecutive_failures: int = 3,
    ):
        self._project_root = project_root
        self._check_interval = check_interval_s
        self._max_failures = max_consecutive_failures
        self._running = False
        self._thread: threading.Thread | None = None
        self._consecutive_failures: int = 0
        self._total_checks: int = 0
        self._last_success_time: float = 0.0

    def start(self):
        """Start the health monitor background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="health-monitor")
        self._thread.start()
        logger.info(f"HealthMonitor started (interval={self._check_interval}s, max_failures={self._max_failures})")

    def stop(self):
        """Stop the health monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        """Main monitoring loop."""
        # Initial delay: let the server settle
        time.sleep(10)
        while self._running:
            try:
                self._probe()
            except Exception as e:
                logger.warning(f"HealthMonitor probe error: {e}")
            time.sleep(self._check_interval)

    def _probe(self):
        """Run one health probe."""
        self._total_checks += 1
        engine = self._get_engine()
        if engine is None:
            self._record_failure("engine_unavailable")
            return

        try:
            raw = engine.wake_up()
            data = json.loads(raw) if isinstance(raw, str) else raw
            status = data.get("status", "unknown")

            if status == "awake":
                self._record_success(data)
            else:
                self._record_failure(f"unexpected_status: {status}")
        except Exception as e:
            self._record_failure(str(e)[:100])

    def _record_success(self, data: dict):
        """Handle a successful probe."""
        self._consecutive_failures = 0
        self._last_success_time = time.monotonic()
        mem = data.get("memory_stats", {})
        logger.debug(
            f"Health: OK (L0={mem.get('l0_evidence',0)}, "
            f"L1={mem.get('l1_nodes',0)}, edges={mem.get('l1_edges',0)})"
        )

    def _record_failure(self, reason: str):
        """Handle a failed probe."""
        self._consecutive_failures += 1
        logger.warning(
            f"Health: FAIL ({self._consecutive_failures}/{self._max_failures}) — {reason}"
        )

        if self._consecutive_failures >= self._max_failures:
            self._alert(reason)

    def _alert(self, reason: str):
        """Push a critical alert to the signal bus."""
        try:
            from cognition.signals import get_signal_bus
            bus = get_signal_bus()
            bus.push(
                "observer_alert",
                priority=10,  # Critical
                payload={
                    "issue": "health_check_failure",
                    "consecutive_failures": self._consecutive_failures,
                    "last_reason": reason,
                    "recommendation": (
                        "FourDMem engine health check failed multiple times. "
                        "Consider restarting the MCP Server or checking the database integrity."
                    ),
                },
            )
            logger.error(f"CRITICAL: Health alert pushed — {reason}")
        except ImportError:
            logger.error(f"CRITICAL: Health check failed but signal bus unavailable — {reason}")

    @property
    def status(self) -> dict:
        """Get current health status."""
        return {
            "running": self._running,
            "consecutive_failures": self._consecutive_failures,
            "total_checks": self._total_checks,
            "last_success_ago_s": (
                time.monotonic() - self._last_success_time
                if self._last_success_time > 0
                else None
            ),
            "healthy": self._consecutive_failures < self._max_failures,
        }

    @staticmethod
    def _get_engine() -> Any:
        try:
            from mcp_server.state import get_engine
            return get_engine()
        except Exception:
            return None


# ── Module-level singleton ─────────────────────────────────────────────────────

_monitor: HealthMonitor | None = None


def start_monitor(project_root: str, **kwargs):
    """Start the global health monitor. Idempotent."""
    global _monitor
    if _monitor is not None:
        return
    _monitor = HealthMonitor(project_root, **kwargs)
    _monitor.start()


def stop_monitor():
    """Stop the global health monitor."""
    global _monitor
    if _monitor is not None:
        _monitor.stop()
        _monitor = None
