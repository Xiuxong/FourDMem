"""Salience Detector — Three-signal fusion salience detection.

Combines three independent signals to determine if content is worth extracting:
1. Keyword patterns (fast, ~0ms) — expanded Chinese + English coverage
2. Embedding semantic density (~10ms) — vector norm as information proxy
3. Syntactic patterns (~0ms) — 主谓宾 fact structure matching

Any single high signal can trigger; combined signals boost confidence.
"""

import re
import math
from typing import Any

# ── Expanded keyword patterns ────────────────────────────────────────────────

DECISION_PATTERNS = [
    r"\bmust\b", r"\bshould\b", r"\bprefer\b", r"\bchoose\b", r"\bdecided\b",
    r"\blet's go with\b", r"\bgoing with\b", r"\bwill use\b",
    r"\bbetter to\b", r"\bbest approach\b", r"\bthe right choice\b",
    r"\bswitch(?:ed|ing)?\b", r"\bmigrat(?:e|ed|ing)\b", r"\breplac(?:e|ed|ing)\b",
    r"必须", r"应该", r"决定", r"选择", r"采用", r"倾向于",
    r"用.*比较好", r"最佳.*方案", r"更合适", r"不如",
    r"改为", r"切换", r"放弃", r"排除", r"迁移到", r"替换为", r"不再用",
    r"升级到", r"确认", r"确定", r"约定",
]

ARCHITECTURE_PATTERNS = [
    r"\barchitecture\b", r"\bdesign pattern\b", r"\brefactor\b",
    r"\btech stack\b", r"\bmigrate\b", r"\btrade-?off\b",
    r"\bapproach\b", r"\bstrategy\b", r"\bframework\b",
    r"\bcomponent\b", r"\bmodule\b", r"\blayer\b", r"\bpipeline\b",
    r"架构", r"设计模式", r"重构", r"技术栈", r"权衡", r"方案",
    r"组件", r"模块", r"分层", r"管道", r"流程", r"体系",
]

BUG_PATTERNS = [
    r"\bbug\b", r"\bfix(?:ed|ing)?\b", r"\berror\b", r"\bbroken\b",
    r"\bcrash\b", r"\bregression\b", r"\broot cause\b",
    r"\bdebug\b", r"\bpanic\b", r"\blifetime\b",
    r"\bworkaround\b", r"\bpatch\b", r"\bhotfix\b",
    r"修复", r"错误", r"崩溃", r"回归", r"根因", r"故障",
    r"异常", r"报错", r"排查", r"定位",
]

MEMORY_PATTERNS = [
    r"\bremember\b", r"\bnote that\b", r"\bimportant\b", r"\bcritical\b",
    r"\bnever\b", r"\balways\b", r"\bavoid\b",
    r"\bkeep in mind\b", r"\bdon't forget\b", r"\bmake sure\b",
    r"\bensure\b", r"\bguarantee\b",
    r"记住", r"注意", r"重要", r"关键", r"别忘了", r"切记",
    r"确保", r"保证", r"明确", r"规定",
]

KNOWLEDGE_PATTERNS = [
    r"\bbecause\b", r"\bthe reason\b", r"\bhow it works\b",
    r"\bin other words\b", r"\bessentially\b", r"\bactually\b",
    r"\bturns out\b", r"\bin fact\b", r"\bbasically\b",
    r"原因是", r"因为", r"本质上", r"也就是说",
    r"实际上是", r"原来是", r"其实", r"换句话说",
]

IMPLEMENTATION_PATTERNS = [
    r"\bimplement\b", r"\bfunction\b", r"\bmethod\b", r"\bclass\b",
    r"\bapi\b", r"\bendpoint\b", r"\bhandler\b",
    r"\binterface\b", r"\bstruct\b", r"\btrait\b",
    r"实现", r"函数", r"方法", r"接口", r"模块",
    r"类", r"结构体", r"特征", r"抽象",
]

CAPABILITY_PATTERNS = [
    r"\bcan\b", r"\bcannot\b", r"\bcapable\b", r"\bsupport\b",
    r"\ballow\b", r"\bprevent\b", r"\benable\b", r"\blimit\b",
    r"\bcompatible\b", r"\brestrict\b",
    r"可以", r"不能", r"无法", r"支持", r"允许",
    r"禁止", r"防止", r"限制", r"兼容", r"能够", r"会",
]

CAUSAL_PATTERNS = [
    r"\bcause\b", r"\bresult\b", r"\blead to\b", r"\bdepend\b",
    r"\btherefore\b", r"\bhence\b", r"\bso that\b",
    r"导致", r"造成", r"使得", r"所以", r"因此",
    r"结果是", r"取决于", r"影响", r"决定",
]

COMPARE_PATTERNS = [
    r"\bfaster\b", r"\bslower\b", r"\bbetter\b", r"\bworse\b",
    r"\bdifferent\b", r"\bsimilar\b", r"\bequivalent\b",
    r"\bcompared\b", r"\bvs\b", r"\bversus\b",
    r"比.*快", r"比.*慢", r"优于", r"劣于",
    r"不同于", r"类似", r"相当于", r"区别在于", r"相比",
]

TIME_PATTERNS = [
    r"\bcurrent\b", r"\bprevious\b", r"\bformerly\b", r"\blater\b",
    r"\balready\b", r"\byet\b", r"\bongoing\b", r"\bwill be\b",
    r"当前", r"之前", r"以后", r"已经", r"尚未",
    r"正在", r"将会", r"曾经", r"目前", r"现在",
]

LOW_SALIENCE_PATTERNS = [
    r"^(hi|hello|hey|ok|thanks|thank you|好的|谢谢|你好)$",
    r"^(yes|no|sure|maybe|是|否|对|好的)$",
    r"^(got it|understood|明白|了解|继续|接着)$",
    r"^(continue|go on|next|继续|下一步)$",
]

_KEYWORD_GROUPS = [
    (MEMORY_PATTERNS, 2.5),
    (DECISION_PATTERNS, 2.0),
    (ARCHITECTURE_PATTERNS, 1.8),
    (BUG_PATTERNS, 1.5),
    (CAUSAL_PATTERNS, 1.3),
    (KNOWLEDGE_PATTERNS, 1.0),
    (CAPABILITY_PATTERNS, 1.0),
    (IMPLEMENTATION_PATTERNS, 0.8),
    (COMPARE_PATTERNS, 0.8),
    (TIME_PATTERNS, 0.5),
]

# ── Syntactic patterns (主谓宾 fact structures) ───────────────────────────────

_SYNTAX_PATTERNS = [
    # X 是 Y / X is Y
    (r'[\u4e00-\u9fff\w]{2,}\s*是\s*[\u4e00-\u9fff\w]{2,}', 1.5),
    (r'\b\w{2,}\s+is\s+\w{2,}', 1.5),
    # X 使用/用 Y / X uses Y
    (r'[\u4e00-\u9fff\w]{2,}\s*(?:使用|用|采用|基于)\s*[\u4e00-\u9fff\w]{2,}', 1.5),
    (r'\b\w{2,}\s+(?:uses?|based on|built with)\s+\w{2,}', 1.5),
    # X 负责 Y / X handles Y
    (r'[\u4e00-\u9fff\w]{2,}\s*(?:负责|处理|管理|控制|实现)\s*[\u4e00-\u9fff\w]{2,}', 1.5),
    (r'\b\w{2,}\s+(?:handles?|manages?|controls?|implements?)\s+\w{2,}', 1.5),
    # X 导致 Y / X causes Y
    (r'[\u4e00-\u9fff\w]{2,}\s*(?:导致|造成|使得|产生)\s*[\u4e00-\u9fff\w]{2,}', 1.5),
    (r'\b\w{2,}\s+(?:causes?|leads? to|results? in)\s+\w{2,}', 1.5),
    # X 支持 Y / X supports Y
    (r'[\u4e00-\u9fff\w]{2,}\s*(?:支持|包含|提供|具备)\s*[\u4e00-\u9fff\w]{2,}', 1.2),
    (r'\b\w{2,}\s+(?:supports?|includes?|provides?|offers?)\s+\w{2,}', 1.2),
    # X 比 Y 快/好 / X is faster/better than Y
    (r'[\u4e00-\u9fff\w]{2,}\s*比\s*[\u4e00-\u9fff\w]{2,}\s*(?:快|慢|好|差|强|弱)', 1.5),
    (r'\b\w{2,}\s+is\s+(?:faster|slower|better|worse)\s+than\s+\w{2,}', 1.5),
    # 数字+单位 (版本号、性能数据)
    (r'\d+\.\d+\s*(?:ms|秒|MB|GB|倍|%)', 1.0),
    (r'\bv?\d+\.\d+(?:\.\d+)?\b', 0.8),
]


class SalienceDetector:
    """Three-signal fusion salience detector.

    Signals:
    1. Keyword patterns (fast, ~0ms) — broad Chinese + English coverage
    2. Embedding semantic density (~10ms) — vector norm as information proxy
    3. Syntactic patterns (~0ms) — 主谓宾 fact structure matching

    Any single high signal can trigger extraction. Combined signals boost
    the score multiplicatively.
    """

    def __init__(self, threshold: float = 1.2):
        self.threshold = threshold
        self._pending_salient: bool = False
        self._pending_content: list[str] = []
        self._embedder = None

    def _get_embedder(self):
        """Lazy-load embedder singleton."""
        if self._embedder is None:
            try:
                from cognition.embedder import get_embedder
                self._embedder = get_embedder()
            except ImportError:
                pass
        return self._embedder

    def check(self, content: str) -> float:
        """Check content salience with three-signal fusion.

        Args:
            content: Text to check.

        Returns:
            Salience score (0.0 = low, higher = more salient).
        """
        if not content or len(content.strip()) < 5:
            return 0.0

        text = content.strip().lower()
        for pattern in LOW_SALIENCE_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                return 0.0

        # Signal 1: Keyword scoring (fast, always runs)
        keyword_score = self._keyword_score(content)

        # Signal 2: Embedding semantic density (if model available)
        semantic_score = self._semantic_density_score(content)

        # Signal 3: Syntactic pattern matching (fast, always runs)
        syntax_score = self._syntax_pattern_score(content)

        # Fusion: weighted sum
        total_score = keyword_score * 1.0 + semantic_score * 0.8 + syntax_score * 1.2

        # Multi-signal bonus: any two signals positive → +0.5
        signals_positive = sum(1 for s in [keyword_score, semantic_score, syntax_score] if s > 0.3)
        if signals_positive >= 2:
            total_score += 0.5
        if signals_positive >= 3:
            total_score += 0.3

        # Buffer if above threshold
        if total_score >= self.threshold:
            self._pending_salient = True
            self._pending_content.append(content[:500])

        return total_score

    def _keyword_score(self, content: str) -> float:
        """Fast keyword-based scoring."""
        score = 0.0
        for patterns, weight in _KEYWORD_GROUPS:
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    score += weight
                    break
        return score

    def _semantic_density_score(self, content: str) -> float:
        """Embedding-based semantic density scoring.

        Uses vector norm as information density proxy.
        Higher norm = more semantic content = more salient.
        """
        emb = self._get_embedder()
        if emb is None or not getattr(emb, '_loaded', False):
            return 0.0

        try:
            vec = emb.embed(content[:500])
            if not vec or all(x == 0.0 for x in vec):
                return 0.0

            # Vector norm as information density
            norm = math.sqrt(sum(x * x for x in vec))

            # Normalize to 0-1 range (typical bge-small norms are 0.5-1.5)
            if norm < 0.5:
                return 0.0
            elif norm < 0.8:
                return 0.3
            elif norm < 1.0:
                return 0.6
            else:
                return 1.0

        except Exception:
            return 0.0

    def _syntax_pattern_score(self, content: str) -> float:
        """Syntactic pattern matching — detect 主谓宾 fact structures."""
        score = 0.0
        for pattern, weight in _SYNTAX_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                score += weight
        return min(score, 3.0)  # cap at 3.0

    def should_extract(self) -> bool:
        """Check if extraction should be triggered."""
        return self._pending_salient

    def get_pending_content(self) -> list[str]:
        """Get buffered salient content and reset state."""
        content = self._pending_content[:]
        self._pending_salient = False
        self._pending_content.clear()
        return content

    def reset(self):
        """Reset detector state."""
        self._pending_salient = False
        self._pending_content.clear()
