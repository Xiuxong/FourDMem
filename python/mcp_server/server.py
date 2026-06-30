"""MCP Server for FourDMem — Agent Memory Core v4.0.

Design philosophy: **Agent uses memory, engine manages it.**

Thin entry point: imports state, lifecycle, and tools from submodules.

Usage:
    python -m mcp_server.server [--db-path PATH]
"""

import argparse
import json
import os
import sys
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("ERROR: 'mcp' package not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# ── System prompt ─────────────────────────────────────────────────────────────

FOURDMEM_SYSTEM_PROMPT = """你有一个自动化的长期记忆系统 FourDMem。

### 核心操作（每次对话必须执行）

1. **会话开始** → 调用 `wake_up` 恢复记忆上下文
2. **回答前** → 调用 `search_memory` 用用户原话搜索记忆
3. **结果不足时** → 换关键词再搜（见下方检索策略）
4. **结果评价** → 检索结果有用调 `submit_feedback` (+1)，无用调 (-1)
5. **每轮结束** → 调用 `extract_deep` 提取本轮关键事实

### 检索策略（Agentic RAG）

如果第一次搜索结果不完整，你应该**主动多轮检索**：

1. **首轮**: `search_memory("用户原话")` — 直接搜索
2. **评估**: 结果是否包含完整答案？
   - ✅ 完整 → 直接回答
   - ⚠️ 部分 → 换关键词再搜
   - ❌ 没找到 → 拆分问题分别搜
3. **二轮**: 用不同关键词搜索
4. **合并**: 综合多轮搜索结果回答

### 提取规则（extract_deep）

每轮对话结束前，调用 `extract_deep` 提取 1-3 条原子事实：
- **提取**: 决策、偏好、事实、技术选型、人员关系
- **跳过**: 闲聊、寒暄、已提取过的重复信息
- **标注 importance**: 关键决策 0.9+，一般事实 0.5-0.8

### 自动归档

系统会自动归档所有工具交互到 L0。你**不需要**手动调用 `log_turn`。

### 进阶工具

- `save_memory` — 用户说"记住这个"时，显式保存关键事实
- `synthesize_l2` — 综合多条记忆形成知识场景
- `reflect_and_synthesize` — 新旧知识冲突时做辩证式综合
- `reflect` — 检索结果不足时评估置信度
- `abandon_branch` — 放弃方案时标记反事实
- `get_entity_context` — 查看记忆的前世今生

### 核心原则

- **每次回答前必须调 search_memory**，结果不足时换关键词再搜
- **每轮结束必须调 extract_deep 提取关键事实**
- 工具交互自动归档，无需手动 log_turn
### 认知信号检查（每轮可选）

系统会自动生成认知信号。建议在对话开始时调用 `check_cognition_signals` 查看是否有待处理的认知任务（髓鞘化候选、范式危机、拓扑相变等）。

- 记忆是透明的底层设施，你负责使用它，不是管理它"""

# ── MCP Server setup ─────────────────────────────────────────────────────────

mcp = FastMCP("FourDMem", instructions=FOURDMEM_SYSTEM_PROMPT)


# ── Tool auto-capture middleware ──────────────────────────────────────────────

import functools
from mcp_server.lifecycle import _auto_archive, _post_interaction

# Read-only tools excluded from auto-capture (no side effects to archive)
_READ_ONLY_TOOLS = frozenset({
    "wake_up", "get_entity_context", "evolution_status",
    "check_cognition_signals", "reload_modules",
})


def _auto_capture_wrapper(fn, tool_name: str):
    """Wrap an MCP tool with automatic L0 audit trail + post-interaction tick.

    Every non-read-only tool invocation gets an audit record in L0.
    _post_interaction is idempotent — safe even if the tool also calls it.
    Tool-specific _auto_archive calls inside the function body provide rich
    content records (user queries, feedback scores, etc.) and are NOT replaced.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if tool_name not in _READ_ONLY_TOOLS:
            try:
                _auto_archive("tool_call", tool_name, {"tool": tool_name})
            except Exception:
                pass

        result = fn(*args, **kwargs)

        if tool_name not in _READ_ONLY_TOOLS:
            try:
                _post_interaction()
            except Exception:
                pass

        return result

    return wrapper

# ── Register tools ───────────────────────────────────────────────────────────

from mcp_server.tools import (
    search_memory as _search_memory,
    submit_feedback as _submit_feedback,
    wake_up as _wake_up,
    log_turn as _log_turn,
    checkpoint_turn as _checkpoint_turn,
    save_memory as _save_memory,
    reflect as _reflect,
    abandon_branch as _abandon_branch,
    re_enable_branch as _re_enable_branch,
    checkpoint_memory as _checkpoint_memory,
    load_memory as _load_memory,
    get_entity_context as _get_entity_context,
    memory_health as _memory_health,
    write_scenario as _write_scenario,
    extract_deep as _extract_deep,
    synthesize_l2 as _synthesize_l2,
    reflect_and_synthesize as _reflect_and_synthesize,
    evolution_status as _evolution_status,
    check_cognition_signals as _check_cognition_signals,
    reload_modules as _reload_modules,
    rebuild_l1 as _rebuild_l1,
    interact as _interact,
)
mcp.tool()(_auto_capture_wrapper(_search_memory, "search_memory"))
mcp.tool()(_auto_capture_wrapper(_submit_feedback, "submit_feedback"))
mcp.tool()(_auto_capture_wrapper(_wake_up, "wake_up"))
mcp.tool()(_auto_capture_wrapper(_log_turn, "log_turn"))
mcp.tool()(_auto_capture_wrapper(_reflect, "reflect"))
mcp.tool()(_auto_capture_wrapper(_abandon_branch, "abandon_branch"))
mcp.tool()(_auto_capture_wrapper(_checkpoint_turn, "checkpoint_turn"))
mcp.tool()(_auto_capture_wrapper(_save_memory, "save_memory"))
mcp.tool()(_auto_capture_wrapper(_re_enable_branch, "re_enable_branch"))
mcp.tool()(_auto_capture_wrapper(_checkpoint_memory, "checkpoint_memory"))
mcp.tool()(_auto_capture_wrapper(_load_memory, "load_memory"))
mcp.tool()(_auto_capture_wrapper(_get_entity_context, "get_entity_context"))
mcp.tool()(_auto_capture_wrapper(_memory_health, "memory_health"))
mcp.tool()(_auto_capture_wrapper(_write_scenario, "write_scenario"))
mcp.tool()(_auto_capture_wrapper(_extract_deep, "extract_deep"))
mcp.tool()(_auto_capture_wrapper(_synthesize_l2, "synthesize_l2"))
mcp.tool()(_auto_capture_wrapper(_reflect_and_synthesize, "reflect_and_synthesize"))
mcp.tool()(_auto_capture_wrapper(_rebuild_l1, "rebuild_l1"))
mcp.tool()(_auto_capture_wrapper(_reload_modules, "reload_modules"))
mcp.tool()(_auto_capture_wrapper(_check_cognition_signals, "check_cognition_signals"))
mcp.tool()(_auto_capture_wrapper(_evolution_status, "evolution_status"))
mcp.tool()(_auto_capture_wrapper(_interact, "interact"))
# ── MCP Resources ─────────────────────────────────────────────────────────────

from mcp_server.state import _session_id, _interaction_count, _turns_since_log, _db_path
from mcp_server.state import _auto_archive_buffer


@mcp.resource("fourdmem://session")
def session_state() -> str:
    """Current session state: turn count, buffered interactions, memory stats."""
    return json.dumps({
        "session_id": _session_id,
        "interaction_count": _interaction_count,
        "buffered_interactions": len(_auto_archive_buffer),
        "db_path": _db_path,
        "auto_capture": "active",
    }, indent=2)


# ── Auto-flush on exit ────────────────────────────────────────────────────────

import atexit
from mcp_server.lifecycle import _flush_on_exit, _start_evolution_scheduler
atexit.register(_flush_on_exit)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import mcp_server.state as state

    parser = argparse.ArgumentParser(description="FourDMem MCP Server")
    parser.add_argument("--db-path", default=None, help="Path to L0 SQLite database")
    parser.add_argument("--session-id", default=None, help="Session ID for auto-archiving")
    parser.add_argument("--workspace", default=None, help="Workspace ID for memory isolation")
    parser.add_argument("--model", default=None, help="Model name (e.g. 'deepseek-v4', 'claude-4')")
    args = parser.parse_args()

    if args.session_id:
        state._session_id = args.session_id
    if args.model:
        state._model_name = args.model
    if args.workspace:
        state._workspace_id = args.workspace
        state._WORKSPACE_DIR = os.path.join(state._PROJECT_ROOT, "data", "workspaces", state._workspace_id)

    if args.db_path:
        state._db_path = args.db_path
    else:
        state._db_path = os.path.join(state._WORKSPACE_DIR, "evidence.db")

    if state.fourdmem is not None:
        db_dir = os.path.dirname(state._db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        os.makedirs(os.path.join(state._PROJECT_ROOT, "data", "cognition"), exist_ok=True)
        # Clean up stale tantivy lock files from previous crash/kill
        _cleaned = state._clean_stale_tantivy_locks(state._db_path)
        if _cleaned:
            print(f"FourDMem: removed {_cleaned} stale tantivy lock(s)", file=sys.stderr)
        state._engine = state.fourdmem.FourDMemEngine(state._db_path)
    else:
        print("WARNING: Running without Rust bindings (mock mode).", file=sys.stderr)

    print(f"FourDMem MCP Server starting...", file=sys.stderr)
    print(f"  Workspace: {state._workspace_id}", file=sys.stderr)
    print(f"  Model: {state._model_name}", file=sys.stderr)
    print(f"  Session: {state._session_id}", file=sys.stderr)
    print(f"  Database: {state._db_path}", file=sys.stderr)

    try:
        from mcp_server.state import _get_embedder
        emb = _get_embedder()
        if emb is not None:
            emb.warmup()
    except Exception:
        pass

    _start_evolution_scheduler()
    mcp.run()


if __name__ == "__main__":
    main()
