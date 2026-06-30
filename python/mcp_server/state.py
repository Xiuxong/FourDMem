"""FourDMem MCP Server — Global state and singletons.

Contains all shared state, singleton factory, and L3 rule management.
Imported by server.py, tools.py, lifecycle.py.
"""

import json
import os
import threading
import uuid
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────

_engine: Any = None
_session_id: str = f"auto-{uuid.uuid4().hex[:8]}"
_interaction_count: int = 0
_turns_since_log: int = 0
_AUTO_CAPTURE_INTERVAL: int = 5
_last_extraction_time: float = 0.0
_EXTRACTION_COOLDOWN: float = 10.0  # seconds between L0→L1 extraction runs
_auto_archive_buffer: list[dict] = []
_l0_content_hashes: set[str] = set()  # content dedup for L0 writes
_has_waken_up: bool = False  # auto-wake on first search_memory call
_l1_loaded: bool = False  # avoid re-loading L1 from disk

_model_name: str = os.environ.get("FOURDMEM_MODEL", "unknown")
_agent_visibility: str = "shared"

_workspace_id: str = os.environ.get(
    "FOURDMEM_WORKSPACE",
    os.path.basename(os.getcwd()).replace(" ", "_").lower(),
)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WORKSPACE_DIR: str = os.path.join(_PROJECT_ROOT, "data", "workspaces", _workspace_id)
_db_path: str = os.path.join(_WORKSPACE_DIR, "evidence.db")

# ── Singleton factory ────────────────────────────────────────────────────────

_singletons: dict[str, Any] = {}


def _get(name: str, import_path: str, cls_name: str, **kwargs) -> Any:
    """Lazy singleton factory. Import once, cache forever."""
    if name not in _singletons:
        try:
            mod = __import__(import_path, fromlist=[cls_name])
            _singletons[name] = getattr(mod, cls_name)(**kwargs)
        except (ImportError, AttributeError) as e:
            logger.debug(f"Singleton {name} unavailable: {e}")
            _singletons[name] = None
    return _singletons[name]


def _get_observer():
    return _get("observer", "evolution.strange_loop", "ObserverNode")

def _get_paradigm_engine():
    return _get("paradigm_engine", "evolution.paradigm_shift", "ParadigmShiftEngine")

def _get_dream_pruner():
    return _get("dream_pruner", "cognition.dream", "DreamPruner")

def _get_myelination():
    return _get("myelination", "evolution.myelination", "MyelinationTracker")

def _get_aggregator():
    return _get("aggregator", "cognition.aggregator", "AutoAggregator", aggregation_threshold=3)

def _get_auto_plugin_generator():
    return _get("auto_plugin", "evolution.auto_plugin", "AutoPluginGenerator")

def _get_analogy_engine():
    return _get("analogy", "evolution.analogy_engine", "AnalogyEngine")

def _get_promoter():
    return _get("promoter", "cognition.promoter", "AutoPromoter", project_root=_PROJECT_ROOT)

def _get_topology():
    return _get("topology", "evolution.topology", "TopologicalMonitor")

def _get_embedder():
    return _get("embedder", "cognition.embedder", "get_embedder") or (
        _singletons.__setitem__("embedder", __import__("cognition.embedder", fromlist=["get_embedder"]).get_embedder())
        or _singletons.get("embedder")
    )

def _get_salience():
    return _get("salience", "cognition.salience", "SalienceDetector", threshold=2.0)


def _get_turn_classifier():
    """Get or create the TurnType classifier singleton."""
    from cognition.turn_classifier import get_turn_classifier
    model_dir = os.path.join(_PROJECT_ROOT, "data", "cognition")
    model_path = os.path.join(model_dir, "turn_classifier.pkl")
    return get_turn_classifier(model_path)

# ── L3 Core Rules Management ────────────────────────────────────────────────

_L3_RULES: list[dict] = []
_L3_LOCK = threading.Lock()


def _load_l3_rules() -> list[dict]:
    """Load L3 core rules from persona.yaml (cached, thread-safe)."""
    global _L3_RULES
    if _L3_RULES:
        return _L3_RULES
    persona_path = os.path.join(_PROJECT_ROOT, "data", "vault", "persona", "persona.yaml")
    if os.path.exists(persona_path):
        try:
            import yaml
            with open(persona_path, "r", encoding="utf-8") as f:
                persona = yaml.safe_load(f) or {}
            with _L3_LOCK:
                _L3_RULES = persona.get("rules", [])
        except Exception:
            pass
    return _L3_RULES


def _save_l3_rule(rule: dict):
    """Append a rule to L3 persona.yaml (thread-safe)."""
    global _L3_RULES
    persona_dir = os.path.join(_PROJECT_ROOT, "data", "vault", "persona")
    persona_path = os.path.join(persona_dir, "persona.yaml")
    os.makedirs(persona_dir, exist_ok=True)
    with _L3_LOCK:
        persona = {}
        if os.path.exists(persona_path):
            try:
                import yaml
                with open(persona_path, "r", encoding="utf-8") as f:
                    persona = yaml.safe_load(f) or {}
            except Exception:
                pass
        if "rules" not in persona:
            persona["rules"] = []
        persona["rules"].append(rule)
        _L3_RULES = persona["rules"]
        try:
            import yaml
            with open(persona_path, "w", encoding="utf-8") as f:
                yaml.dump(persona, f, allow_unicode=True, default_flow_style=False)
        except Exception:
            pass


# ── Engine singleton ─────────────────────────────────────────────────────────

try:
    import fourdmem
except ImportError:
    fourdmem = None


def _clean_stale_tantivy_locks(db_path: str) -> int:
    """Remove .tantivy-*.lock files left by crashed process. Safe no-op if none."""
    import glob as _glob
    _fulltext_dir = os.path.join(os.path.dirname(db_path), "fulltext")
    _removed = 0
    for _lock in _glob.glob(os.path.join(_fulltext_dir, ".tantivy-*.lock")):
        try:
            os.remove(_lock)
            _removed += 1
        except OSError:
            pass
    return _removed


def get_engine():
    """Get or create the FourDMem engine singleton."""
    global _engine
    if _engine is None:
        if fourdmem is not None:
            db_dir = os.path.dirname(_db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            _clean_stale_tantivy_locks(_db_path)
            _engine = fourdmem.FourDMemEngine(_db_path)
        else:
            raise RuntimeError("FourDMem Rust bindings not available.")
    return _engine
