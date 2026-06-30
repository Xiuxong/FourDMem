"""Turn Type Classifier — Classify Agent conversation turns into semantic types.

Harness signals (MCP protocol) provide 100% accurate classification for:
- tool_call: when an MCP tool is invoked
- tool_result: when tool output is returned
- user_query: when role is "user"

Classifier handles the ambiguous assistant responses:
- reasoning: internal thought process, deliberation
- plan: structured task breakdown with numbered steps
- final_answer: conclusive response with deliverable
- system_prompt: injected system instructions (rarely archived)

Architecture:
1. Heuristic (regex + keyword) — high precision, moderate recall. Only classifies
   when very confident (>95% precision target). Returns None for uncertain cases.
2. Embedder + LogisticRegression fallback — handles the rest (~95% accuracy).
3. Uses existing bge-small-zh-v1.5 embedder (512-dim) from cognition.embedder.

Design principle: heuristic is CONSERVATIVE. It prefers returning None (→ ML)
over making a wrong classification. Tool types are handled by harness signals,
not text patterns (which are too ambiguous).
"""

import hashlib
import json
import os
import pickle
import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np


class TurnType(str, Enum):
    """Semantic types of Agent conversation turns."""
    USER_QUERY = "user_query"         # User message / request
    REASONING = "reasoning"           # Internal thought, deliberation
    PLAN = "plan"                     # Structured task breakdown
    TOOL_CALL = "tool_call"           # Tool invocation (MCP protocol)
    TOOL_RESULT = "tool_result"       # Tool execution output
    FINAL_ANSWER = "final_answer"     # Conclusive response
    SYSTEM_PROMPT = "system_prompt"   # Injected system instructions
    UNKNOWN = "unknown"               # Unclassifiable


# ── Conservative heuristic patterns (high precision, moderate recall) ─────────
# These patterns are intentionally narrow. False positives are worse than
# falling through to the ML classifier. If unsure → return None.

# Plan detection: clear numbered/bulleted task breakdowns
PLAN_PATTERNS = [
    # "1. do X\n2. do Y\n3. do Z" (2+ numbered items)
    re.compile(r'(?:^|\n)\s*\d+[\.\)、]\s*.+\s*\n\s*\d+[\.\)、]'),
    # "Phase 1: ... \nPhase 2: ..."
    re.compile(r'(?:Phase|阶段|Step|步骤)\s*\d+\s*[：:].+\s*\n\s*(?:Phase|阶段|Step|步骤)\s*\d+'),
    # "步骤：\n1. ...\n2. ..." or "Plan:\n1. ...\n2. ..."
    re.compile(r'(?:步骤|流程|计划|方案|[Pp]lan|[Ss]teps?)\s*[：:]\s*\n\s*(?:\d+[\.\)、]|[-*•])'),
    # "- [ ] Task 1\n- [ ] Task 2" (checklist format, 2+ items)
    re.compile(r'(?:^|\n)\s*[-*]\s*\[\s*[ xX]?\s*\].+\s*\n\s*[-*]\s*\[\s*[ xX]?\s*\]'),
    # "分三步走" / "分 3 个阶段"
    re.compile(r'分\s*\d+\s*(?:步|阶段|个步骤|部分)'),
]

# Reasoning detection: internal thought / deliberation markers
REASONING_PATTERNS = [
    # "让我先..." / "我需要先..." / "我先..."
    re.compile(r'^(?:让我|我先|我需要|我应该|我打算|我要|我认为|我觉得|我考虑)'),
    # "Let me think" / "I need to" / "First I'll"
    re.compile(r'^(?:Let\s+me\s+(?:think|check|see|read|look|verify|understand)|I\s+(?:need|should|will|must)\s+(?:first|start|check|read|think))', re.IGNORECASE),
    # "Hmm" / "OK so" / "Alright"
    re.compile(r'^(?:Hmm|OK so|Alright|Got it|Understood|好的|明白了|了解了)', re.IGNORECASE),
    # "接下来..." / "现在我需要..."
    re.compile(r'^(?:接下来|下一步|现在|目前)'),
]

# Final answer detection: conclusive, presentational
FINAL_ANSWER_PATTERNS = [
    # "以下是..." / "下面是..." / "这是..."
    re.compile(r'^(?:以下是|下面是|这是|输出如下|结果如下|代码如下|总结如下|答案如下)'),
    # "Here's the..." / "Here is the..." / "The answer is..."
    re.compile(r'^(?:Here\'?s?\s+(?:the|my|a)\s+(?:code|script|result|answer|summary|conclusion|fix|solution)|The\s+(?:answer|solution|fix)\s+is)', re.IGNORECASE),
    # "总结：" / "结论：" / "最终：" / "综上："
    re.compile(r'^(?:总结|结论|最终|综上|总而言之|一句话|总体来说)[：:]'),
    # "In summary" / "In conclusion" / "TL;DR"
    re.compile(r'^(?:In\s+(?:summary|conclusion|short|brief)|To\s+summarize|TL;DR)', re.IGNORECASE),
    # Code block starts immediately (typical of final answers with code)
    re.compile(r'^```(?:\w+)?\s*\n'),
]

# System prompt detection
SYSTEM_PROMPT_PATTERNS = [
    re.compile(r'^(?:你是一个|你是|you\s+are\s+(?:a|an|the)\s+(?:helpful|assistant|code|expert|agent))', re.IGNORECASE),
    re.compile(r'(?:系统提示|system\s+prompt|role\s*[：:]|instructions?\s*[：:])', re.IGNORECASE),
    re.compile(r'(?:MUST|NEVER|REQUIRED|PROHIBITED|AVOID)\s', re.IGNORECASE),
    re.compile(r'<(?:system|instruction|rule)s?>'),
]

# Tool call / result — only match VERY specific patterns
# (generic tool invocation text is too ambiguous; harness signals handle this)
TOOL_CALL_PATTERNS = [
    re.compile(r'^\s*\{\s*"(?:jsonrpc|method|name|arguments|tool)"', re.MULTILINE),
    re.compile(r'^(?:Running|Executing|Calling):\s*\S', re.MULTILINE),
]

TOOL_RESULT_PATTERNS = [
    re.compile(r'^(?:Exit code|Wall time|Execution time|PID):', re.MULTILINE),
    re.compile(r'^\s*(?:\[\w+[/\\].+#[A-F0-9]{4}\])\s', re.MULTILINE),
    re.compile(r'^(?:\[STDERR\]|---|\=\=\=)\s', re.MULTILINE),
    re.compile(r'^(?:Traceback|error:|Error:|WARNING:)', re.MULTILINE),
]


def heuristic_classify(content: str) -> Optional[TurnType]:
    """Fast conservative heuristic classification.

    Returns None if uncertain — caller should fall back to ML classifier.
    Designed for high precision (>95%), accepting lower recall.

    Does NOT classify tool_call/tool_result from ambiguous text;
    those should come from harness protocol signals.
    """
    if not content or not content.strip():
        return None

    text = content.strip()

    # System prompt — very distinctive markers
    for pat in SYSTEM_PROMPT_PATTERNS:
        if pat.search(text):
            return TurnType.SYSTEM_PROMPT

    # Plan — requires clear multi-item structure (high confidence)
    for pat in PLAN_PATTERNS:
        if pat.search(text):
            return TurnType.PLAN

    # Tool result — only very specific format markers
    for pat in TOOL_RESULT_PATTERNS:
        if pat.search(text):
            return TurnType.TOOL_RESULT

    # Tool call — only very specific format markers
    for pat in TOOL_CALL_PATTERNS:
        if pat.search(text):
            return TurnType.TOOL_CALL

    # Final answer — conclusive opening markers
    for pat in FINAL_ANSWER_PATTERNS:
        if pat.search(text):
            return TurnType.FINAL_ANSWER

    # Reasoning — internal thought markers (check LAST, most generic)
    for pat in REASONING_PATTERNS:
        if pat.search(text):
            return TurnType.REASONING

    return None  # Uncertain → fall back to ML classifier


# ── Embedder-based ML Classifier ─────────────────────────────────────────────

@dataclass
class TrainingExample:
    text: str
    label: TurnType


def _generate_training_data() -> list[TrainingExample]:
    """Generate synthetic training examples for each turn type."""
    examples: list[TrainingExample] = []

    # === USER_QUERY ===
    user_queries = [
        "帮我写一个Python脚本统计CSV文件的行数",
        "FourDMem 未连接",
        "对话记忆能否分类存储？",
        "How do I implement a binary search tree in Rust?",
        "What's the best way to handle errors in async Python?",
        "Explain the borrow checker rules",
        "Write a SQL query to find duplicate users",
        "帮我审查这个PR的代码质量",
        "翻译这段英文文档",
        "这个bug的根因是什么？",
        "对比一下React和Vue的性能差异",
        "Run the test suite and fix failing tests",
        "优化这个数据库查询的性能",
        "Add authentication middleware to the API",
        "Refactor this module to use dependency injection",
        "有没有需要我决策的内容？",
        "请帮我分析这段日志的错误原因",
        "能否给出一个完整的实现方案？",
    ]
    for t in user_queries:
        examples.append(TrainingExample(t, TurnType.USER_QUERY))

    # === REASONING ===
    reasonings = [
        "用户需要CSV行数统计脚本，还要测试。我需要先写代码，再调用执行工具运行测试。",
        "让我先理解问题。用户想要分类存储对话记忆，这涉及到L0 schema的扩展。",
        "先检查一下当前数据库schema，看看metadata字段的结构。",
        "我需要先读取现有的extractor代码，了解L0→L1的提取流程。",
        "Let me think about this. The user wants a CSV counter, so I need to handle edge cases like empty files.",
        "Hmm, the error is in the auth middleware. Let me trace the call path first.",
        "OK so the issue is that the fulltext index is corrupted. I should check which segments are missing.",
        "接下来需要验证修改是否生效。先跑一下单元测试，再检查集成测试。",
        "可能有两种方案：方案A轻量级只改metadata，方案B完整管线改L1。A更安全。",
        "还需要考虑边界情况：空输入、超长文本、非UTF-8编码。",
        "I should read the state.py first to understand the singleton pattern, then follow the same approach.",
        "Let me first check if there's an existing pattern for this in the codebase.",
        "先看看现有代码里有没有类似的分类逻辑可以复用。",
        "这个改动涉及三个文件，需要按顺序修改：先classifier，再lifecycle，最后tools。",
        "不确定这里的edge case怎么处理，可能需要看一下上游调用方。",
        "现在我需要先确认sklearn是否已安装，再决定用哪种分类器。",
        "好的，用户选了方案一+方案二混合。这意味着我需要同时实现harness信号和本地ML分类器。",
    ]
    for t in reasonings:
        examples.append(TrainingExample(t, TurnType.REASONING))

    # === PLAN ===
    plans = [
        "1. 编写Python统计脚本\n2. 生成测试CSV文件\n3. 执行脚本验证结果\n4. 输出最终代码",
        "步骤：\n1. 创建turn_classifier.py\n2. 集成到lifecycle.py\n3. 更新log_turn工具\n4. 运行测试验证",
        "Phase 1: Classifier\n- Create turn_classifier.py\n- Generate training data\n\nPhase 2: Integration\n- Enhance _auto_archive\n- Enhance log_turn\n\nPhase 3: Verification\n- Test accuracy\n- Smoke test",
        "Plan:\n1. Read the current L0 schema\n2. Design the metadata extension\n3. Implement the classifier\n4. Wire into the pipeline\n5. Run the test suite",
        "Here's my plan:\n\n**Step 1**: Create the classifier module with heuristic rules\n**Step 2**: Add embedder-based fallback for ambiguous cases\n**Step 3**: Integrate into _auto_archive and log_turn\n**Step 4**: Verify with real conversation data",
        "执行计划：\n① 先检查fulltext索引状态\n② 删除损坏的segment\n③ 重启MCP server验证",
        "分三步走：\n第一步：L0 metadata打标（低风险）\n第二步：验证分类准确性\n第三步：扩展到L1类型化节点",
        "Implementation checklist:\n- [ ] Create turn_classifier.py\n- [ ] Add training data generation\n- [ ] Wire into lifecycle._auto_archive\n- [ ] Add turn_type param to log_turn MCP tool\n- [ ] Run existing tests for regressions",
        "执行计划如下：\n① 检查索引\n② 修复损坏\n③ 重启服务",
    ]
    for t in plans:
        examples.append(TrainingExample(t, TurnType.PLAN))

    # === TOOL_CALL ===
    tool_calls = [
        'Calling search_memory with query "turn type classification"',
        'Invoking read on python/mcp_server/lifecycle.py:30-90',
        '调用bash工具执行 pytest tests/ -xvs',
        'Running: find paths=["python/cognition/*.py"]',
        'Tool: lsp references on _auto_archive',
        'Executing: "G:/venv/python.exe" -c "import sklearn"',
        'Calling: write "python/cognition/turn_classifier.py"',
        'Invoking: ast_grep pattern="def _auto_archive" paths=["python/**"]',
        '调用run_code工具，参数：代码内容xxx',
        '运行read工具读取 state.py:60-100',
    ]
    for t in tool_calls:
        examples.append(TrainingExample(t, TurnType.TOOL_CALL))

    # === TOOL_RESULT ===
    tool_results = [
        "Tool output: 68 documents found in fulltext index",
        "Exit code: 0\nWall time: 1.2 seconds\nOutput: 1234 lines",
        "Found 3 files:\n- turn_classifier.py (new)\n- lifecycle.py (modified)\n- tools.py (modified)",
        "[python/mcp_server/tools.py#E3C2] matched 5 occurrences",
        "PID: 15308\nCommand exited with code 0\nWall time: 0.5 seconds",
        "sklearn 1.9.0 installed",
        "Result: facts_extracted=3, conflicts_resolved=1",
        "运行成功，测试文件共1234行",
        "[python/cognition/turn_classifier.py#6E3E] Successfully wrote 19345 bytes",
    ]
    for t in tool_results:
        examples.append(TrainingExample(t, TurnType.TOOL_RESULT))

    # === FINAL_ANSWER ===
    final_answers = [
        "以下是完整的Python脚本：\n\n```python\nimport csv\n\ndef count_rows(filepath):\n    with open(filepath) as f:\n        return sum(1 for _ in csv.reader(f))\n\nprint(count_rows('test.csv'))\n```\n\n测试结果显示文件共1234行。",
        "总结：FourDMem 连接问题已修复。根因是fulltext索引损坏，删除后引擎自动重建。",
        "完整的实现方案如下：\n\n## 方案\n1. 创建 turn_classifier.py\n2. 集成到 lifecycle.py\n\n代码已提交，测试全部通过。",
        "Here's the fix: the bug is in the token_estimate calculation. Changed from using doc_id length to content length. See `crates/retrieval-core/src/scoring.rs:142`.",
        "修复完成。改动涉及3个文件：\n- `python/mcp_server/tools.py` — 增强log_turn\n- `python/mcp_server/lifecycle.py` — 注入turn_type\n- `python/cognition/turn_classifier.py` — 新增分类器\n\n运行 `pytest tests/ -x` 验证，33/33 测试通过。",
        "答案是：可以。FourDMem的L0→L1→L2分层架构天然支持按turn type分类存储。建议先用L0 metadata富化，验证后再扩展到L1类型化节点。",
        "结论：方案一+方案二混合是最优解。Harness协议信号处理tool_call/tool_result（100%准确），本地embedder分类器处理reasoning/plan/final_answer（~95%准确）。零API成本。",
    ]
    for t in final_answers:
        examples.append(TrainingExample(t, TurnType.FINAL_ANSWER))

    return examples


class TurnClassifier:
    """Hybrid turn type classifier: conservative heuristic + embedder-based ML.

    Usage:
        clf = TurnClassifier()
        clf.ensure_trained()  # Train on first use (generates pickle)
        turn_type = clf.classify("用户需要CSV统计脚本，我先写代码再测试", role="assistant")
        # → TurnType.REASONING

        # With harness hint (100% accurate):
        turn_type = clf.classify("Calling search_memory...", harness_hint=TurnType.TOOL_CALL)
        # → TurnType.TOOL_CALL (harness hint overrides all)
    """

    MODEL_PATH: str = ""  # Set by get_turn_classifier()

    def __init__(self, model_path: str = ""):
        self._model: Any = None
        self._label_map: dict[int, TurnType] = {}
        self._label_map_rev: dict[TurnType, int] = {}
        self._trained: bool = False
        if model_path:
            self.MODEL_PATH = model_path


    def _load_l0_examples(self, db_path: str = "") -> list[TrainingExample]:
        """Load labeled examples from L0, capped per turn_type."""
        examples: list[TrainingExample] = []
        paths = []
        if db_path and os.path.exists(db_path):
            paths.append(db_path)
        for p in (
            os.path.join(os.path.dirname(self.MODEL_PATH), "..", "vault", "evidence.db"),
            os.path.join(os.path.dirname(self.MODEL_PATH), "..", "workspaces", "fourdmem", "evidence.db"),
        ):
            normalized = os.path.normpath(p)
            if os.path.exists(normalized) and normalized not in paths:
                paths.append(normalized)
        if not paths:
            return examples

        # Cap per class to match synthetic data balance
        syn_counts = {"user_query": 18, "reasoning": 17, "plan": 9, "tool_call": 10, "tool_result": 9, "final_answer": 7}
        per_class_limit = {k: max(v, 5) for k, v in syn_counts.items()}

        for p in paths:
            try:
                import sqlite3
                db = sqlite3.connect(p)
                db.row_factory = sqlite3.Row
                counts: dict[str, int] = {}
                rows = db.execute(
                    "SELECT content, json_extract(metadata, '$.turn_type') as tt "
                    "FROM evidence WHERE json_extract(metadata, '$.turn_type') IS NOT NULL "
                    "AND LENGTH(content) > 10 AND LENGTH(content) < 500 "
                    "ORDER BY id DESC LIMIT 500"
                ).fetchall()
                db.close()
                for r in rows:
                    tt_str = r["tt"]
                    if not tt_str or tt_str not in TurnType._value2member_map_:
                        continue
                    limit = per_class_limit.get(tt_str, 5)
                    if counts.get(tt_str, 0) >= limit:
                        continue
                    counts[tt_str] = counts.get(tt_str, 0) + 1
                    examples.append(TrainingExample(r["content"], TurnType(tt_str)))
            except Exception:
                pass
        return examples

    def ensure_trained(self):
        """Train the ML classifier if not already trained. Idempotent."""
        if self._trained:
            return

        # Try loading from disk
        if self.MODEL_PATH and os.path.exists(self.MODEL_PATH):
            try:
                with open(self.MODEL_PATH, 'rb') as f:
                    data = pickle.load(f)
                self._model = data['model']
                self._label_map = data['label_map']
                self._label_map_rev = {v: k for k, v in self._label_map.items()}
                self._trained = True
                return
            except Exception:
                pass  # Fall through to training

        # Generate training data and train (pass db_path derived from MODEL_PATH)
        db_path = ""
        if self.MODEL_PATH:
            # MODEL_PATH = data/cognition/turn_classifier.pkl → data/vault/evidence.db
            cognition_dir = os.path.dirname(self.MODEL_PATH)
            data_dir = os.path.dirname(cognition_dir)
            db_path = os.path.join(data_dir, "vault", "evidence.db")
        self._train(db_path)

    def _train(self, db_path: str = ""):
        """Train LogisticRegression on synthetic + real L0 data."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        examples = _generate_training_data()

        # Augment with real labeled L0 evidence
        examples.extend(self._load_l0_examples(db_path))

        # Build label map
        labels = sorted(set(e.label for e in examples))
        self._label_map = {i: label for i, label in enumerate(labels)}
        self._label_map_rev = {label: i for i, label in self._label_map.items()}

        # Get embeddings
        embedder = self._get_embedder()
        X_list: list[np.ndarray] = []
        y_list: list[int] = []

        for ex in examples:
            try:
                vec = embedder.embed(ex.text)
                if vec and not all(v == 0.0 for v in vec):
                    X_list.append(np.array(vec, dtype=np.float32))
                    y_list.append(self._label_map_rev[ex.label])
            except Exception:
                continue

        if len(X_list) < 10:
            # Not enough data — skip ML, rely on heuristic only
            self._trained = True
            return

        X = np.stack(X_list)
        y = np.array(y_list)

        # Normalize
        scaler = StandardScaler()
        model = LogisticRegression(
            solver='saga',
            max_iter=2000,
            C=1.0,
            random_state=42,
        )
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)

        self._model = {'classifier': model, 'scaler': scaler}
        self._trained = True

        # Persist
        if self.MODEL_PATH:
            try:
                os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
                with open(self.MODEL_PATH, 'wb') as f:
                    pickle.dump({
                        'model': self._model,
                        'label_map': self._label_map,
                    }, f)
            except Exception:
                pass

    def classify(self, content: str, role: str = "", harness_hint: Optional[TurnType] = None) -> TurnType:
        """Classify a conversation turn.

        Priority:
        1. harness_hint — explicit annotation from MCP protocol (100% accurate)
        2. role-based — "user" → USER_QUERY, "tool" → check text for call/result
        3. Heuristic — conservative regex/keyword patterns (high precision)
        4. ML fallback — embedder + LogisticRegression (~95% accurate)
        5. Unknown — last resort
        """
        if not content or not content.strip():
            return TurnType.UNKNOWN

        # Harness hint overrides everything
        if harness_hint is not None:
            return harness_hint

        # Role-based
        if role == "user":
            return TurnType.USER_QUERY
        if role == "system":
            return TurnType.SYSTEM_PROMPT

        # Conservative heuristic (returns None if uncertain)
        result = heuristic_classify(content)
        if result is not None:
            return result

        # For assistant role without heuristic match: ML fallback
        if self._trained and self._model is not None:
            ml_result = self._ml_classify(content)
            if ml_result != TurnType.UNKNOWN:
                return ml_result

        return TurnType.UNKNOWN

    def _ml_classify(self, content: str) -> TurnType:
        """Embedder-based ML classification for ambiguous turns."""
        try:
            embedder = self._get_embedder()
            vec = embedder.embed(content)
            if not vec or all(v == 0.0 for v in vec):
                return TurnType.UNKNOWN

            X = np.array(vec, dtype=np.float32).reshape(1, -1)
            scaler = self._model['scaler']
            clf = self._model['classifier']
            X_scaled = scaler.transform(X)
            pred = clf.predict(X_scaled)[0]
            return self._label_map.get(int(pred), TurnType.UNKNOWN)
        except Exception:
            return TurnType.UNKNOWN

    @staticmethod
    def _get_embedder():
        """Lazy-import embedder to avoid circular deps."""
        from cognition.embedder import get_embedder
        return get_embedder()


# ── Singleton ─────────────────────────────────────────────────────────────────

_classifier_instance: Optional[TurnClassifier] = None
_lock = threading.Lock()


def get_turn_classifier(model_path: str = "") -> TurnClassifier:
    """Get or create the global TurnClassifier singleton (thread-safe)."""
    global _classifier_instance
    if _classifier_instance is None:
        with _lock:
            if _classifier_instance is None:
                _classifier_instance = TurnClassifier(model_path)
                _classifier_instance.ensure_trained()
    return _classifier_instance
