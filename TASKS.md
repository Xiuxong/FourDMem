# 🧠 FourDMem v4.1 — Agent-driven Cognitive Memory Infrastructure

> **版本**: v4.1 (Agent-Driven · 从"自我进化"到"为 Agent 提供认知信号")
> **更新日期**: 2026-06-26
> **核心转变**: FourDMem 从"自己做认知的硅基大脑"重新定位为"Agent 的记忆基础设施 + 认知信号系统"。认知执行（事实提取、辩证综合、类比生成、插件编写）由 Agent 的 LLM 完成；FourDMem 负责存储、检索、排序、时间建模、演化监控。

---

## 📌 架构原则 (v4.1)

### 核心分工

```
┌──────────────────────────────────────────────────────┐
│            Agent (LLM — 认知执行者)                    │
│  · 事实提取 (extract_deep)                             │
│  · 辩证综合 (reflect_and_synthesize)                   │
│  · 跨域类比 (cognition_task)                           │
│  · 插件生成 (cognition_task)                           │
│  · 范式转移决策                                        │
└──────────────────────┬───────────────────────────────┘
                       │ MCP 工具
┌──────────────────────▼───────────────────────────────┐
│         FourDMem (记忆基础设施 + 认知信号)              │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │
│  │ L0-L3 存储  │  │ RIF-U 排序 │  │ 认知信号总线    │  │
│  │ RRF 融合   │  │ 主观时间   │  │ · 髓鞘化候选   │  │
│  │ VersionTree│  │ Token 配额 │  │ · 范式危机告警 │  │
│  │ SSGM 治理  │  │ 渐进披露   │  │ · 拓扑相变信号 │  │
│  └────────────┘  └────────────┘  └────────────────┘  │
└──────────────────────────────────────────────────────┘
```

1. **Agent 拥有认知** — LLM 提取、LLM 辩证、LLM 类比、LLM 写插件。FourDMem 不做规则伪认知。
2. **FourDMem 拥有记忆** — L0-L3 存储、向量/全文/图检索、RIF-U 排序、主观时间、版本树、SSGM 治理。
3. **信号桥接两者** — FourDMem 监控记忆状态，推送信号给 Agent。Agent 决定是否/何时响应。
4. **白盒明文** — L0(SQLite) + L1(JSON) + L2(MD) + L3(YAML)。全部人类可读、可 Git 追踪。
5. **主观时间 (Active Ticks)** — 记忆衰减基于交互次数，非物理时钟。解决长期停机"断崖式失忆"。
6. **渐进披露 + Token 配额** — L3(20%) + L2(40%) + L1(30%) + L0(10%)，元认知自动下钻。
7. **代码严谨性** — 每个模块职责单一。认知执行在 Agent 侧，存储检索在 Rust 侧。无伪实现。

### L0-L4 分层 (v4.1 重定义)

| 层级 | 名称 | 职责 | 谁写入 |
|:---:|:---|:---|:---|
| **L4** | **认知信号层** | 监控记忆状态，生成信号推送给 Agent。不修改自身权重。 | FourDMem (自动) |
| **L3** | 核心画像/规则 | 全局共享的稳定规范。极少变动。 | Agent (通过认知任务) |
| **L2** | 场景知识块 | 条件化规则、跨域类比、叙事记忆。Markdown。 | Agent (LLM 生成) |
| **L1** | 原子事实图谱 | 结构化事实，版本树，效用锚点。 | Agent (extract_deep) |
| **L0** | 原始证据 | 对话 + 工具调用日志。只追加，不可变。 | FourDMem (自动归档) |

---

## Phase 1: 地基重构 — Agent-driven 认知管线

**目标**: 消除所有规则伪认知，建立干净的 Agent ↔ FourDMem 分工边界。

| # | 任务 | 做法 | 验收 |
|:---|:---|:---|:---|
| 1.1 | **删除 extractor.py 规则提取** | 删除 `_extract_rule_based()` (40+ 硬编码关键词)、`merge_learned_indicators()`、`record_indicator_feedback()`、`_agent_refine_facts()`。保留 `extract_from_session()` 但改为从 L0 取证据返回给 Agent 处理。 | `grep _extract_rule_based` 返回空 |
| 1.2 | **Agent-driven extract_deep** | `extract_deep(facts)` 已接受 Agent JSON。增强 docstring 说明 Agent 如何用 LLM 提取事实并提交。Agent 端的认知任务：分析会话 → 提取 1-3 条原子事实 → 标注 importance → 调用 extract_deep。 | Agent 提交的事实有正确的 importance/tags |
| 1.3 | **删除 paradigm_shift.py 伪辩证** | 删除 `_rule_based_dialectic()`（字符串拼接）。保留 `record_outcome()` 监控和 `check_crisis()` 告警。辩证综合由 Agent 通过 `reflect_and_synthesize(domain, thesis, antithesis, synthesis)` 完成。 | `_rule_based_dialectic` 符号不存在 |
| 1.4 | **删除 analogy_engine.py 伪类比** | 删除 `_rule_based_analogy()`（关键词交集匹配）。保留 `_extract_clusters()` 图聚类提取。类比生成由 Agent 的 LLM 完成。 | `_rule_based_analogy` 符号不存在 |
| 1.5 | **删除 auto_plugin.py 模板填充** | 删除 `PLUGIN_TEMPLATE` 和基于模板的生成逻辑。保留 `AutoPluginGenerator` 的痛点检测和沙盒验证框架。插件代码由 Agent 的 LLM 生成。 | 无模板代码残留 |
| 1.6 | **checkpoint_turn 使用 agent-driven 路径** | `checkpoint_turn` 中的 `_bg_extract` 不再调用 `FactExtractor.extract_from_session()` 的规则提取。改为检查 Agent 是否已调用 `extract_deep`，若未调用则推送 `extraction_suggested` 信号。 | 自动提取不再使用关键词规则 |

---

## Phase 2: 认知信号系统

**目标**: FourDMem 从被动查询变为主动信号推送。Agent 在每轮交互后检查信号。

| # | 任务 | 做法 | 验收 |
|:---|:---|:---|:---|
| 2.1 | **SignalBus 基础** | `cognition/signals.py` — 统一信号队列。`push(signal_type, payload)` / `poll()` / `ack(signal_id)`。线程安全。 | 信号入队出队正确 |
| 2.2 | **MCP Tool: check_cognition_signals** | Agent 可调用此工具查看待处理信号。返回信号类型、优先级、上下文。 | Agent 能看到信号 |
| 2.3 | **髓鞘化信号** | 当 `MyelinationTracker` 检测到 `hits >= 10 && success_rate >= 0.9`，推送 `macro_candidate` 信号而非自动编译。 | 信号入队，不自动改状态 |
| 2.4 | **范式转移信号** | 当 `ParadigmShiftEngine.check_crisis()` 返回 True，推送 `paradigm_crisis` 信号（含 domain, failure_rate, old_l3, new_l0）。 | 危机信号正确入队 |
| 2.5 | **拓扑相变信号** | 当 `TopologicalMonitor.check_phase_transition()` 触发，推送 `phase_transition` 信号（含 metrics, clusters）。 | 相变信号正确入队 |
| 2.6 | **梦境修剪信号** | 每 100 ticks，推送 `dream_pruning` 信号（含衰减候选列表）。Agent 决定保留/衰减。 | 修剪信号在 tick 触发 |
| 2.7 | **信号限流与合并** | 同类型信号有冷却时间。同 domain 的危机信号合并。防止信号洪泛。 | 1 分钟内同类型信号不重复 |

---

## Phase 3: 检索增强 — Benchmark 与消融

**目标**: FourDMem 有公开可复现的 benchmark 数据。

| # | 任务 | 做法 | 验收 |
|:---|:---|:---|:---|
| 3.1 | **修复 MCP Server 启动** | 在 `pyproject.toml` 中声明所有 Python 子包，确保 `maturin develop` 正确安装。 | `make run-mcp` 成功 |
| 3.2 | **synthetic_recall benchmark** | 跑通 20 事实 + 18 问题的基准。输出 R@1, R@3, R@5, MRR。 | R@1 ≥ 0.8 |
| 3.3 | **LoCoMo benchmark** | 接入真实 LoCoMo 数据集。LLM-as-Judge 评估。 | 完整 benchmark 报告 |
| 3.4 | **消融实验** | 裸向量 vs +RIF-U vs +RRF vs +SubjectiveTime vs 完整 FourDMem。 | 各组件贡献量化 |
| 3.5 | **公开结果** | `benchmarks/RESULTS.md`。可复现脚本 + 固定种子。 | 任何人可复现 |

---

## Phase 4: 自动化运维

| # | 任务 | 做法 | 验收 |
|:---|:---|:---|:---|
| 4.1 | **CI 启用 Python 测试** | 取消注释 `ci.yml` Python 测试步骤。加 `maturin develop`。 | PR 自动跑全量测试 |
| 4.2 | **进化调度器** | `daemon/evolution_scheduler.py` — 后台线程，定时触发信号检查、梦境修剪、L1→L2 聚合。 | 进化调度器随 MCP Server 启动 |
| 4.3 | **跨层引用完整性** | `integrity_checker.py` 扫描 L1→L0, L2→L1 溯源链完整性。CI 中每日运行。 | 无 CRITICAL 断裂 |
| 4.4 | **健康监控** | 定期 `wake_up` 探针。连续失败自动重启。 | 崩溃 10s 内恢复 |

---

## 💡 附录

### 关键设计决策：为什么 Agent 做认知而不是 FourDMem

| 决策 | 理由 |
|:---|:---|
| **认知由 Agent LLM 执行** | Agent 已有 LLM。在 MCP Server 内嵌 LLM 调用会引入 API key 管理、网络依赖、token 成本不可控。Agent-in-the-loop 更简单、更可控。 |
| **FourDMem 做信号而非决策** | 自动修改自身权重（怪圈自指）在没有充分安全边界的情况下是危险的。改为信号推送，Agent 作为人类可审计的决策者。 |
| **删除规则伪认知** | `_extract_rule_based()` 用 40+ 硬编码关键词提取事实，质量远低于 LLM。保留它会给用户虚假的信心。宁可少做，不做假的。 |
| **保留监控基础设施** | `MyelinationTracker`, `ParadigmShiftEngine`, `TopologicalMonitor` 的监控逻辑是正确的——它们检测模式。只是响应方式从"自动修改"改为"推送信号"。 |

### 与 v4.0 README 的对应关系

| v4.0 README 描述 | v4.1 实现方式 |
|:---|:---|
| "认知髓鞘化 → 直觉" | FourDMem 监控高频成功路径 → 推送 `macro_candidate` 信号 → Agent 决定是否编译为宏 |
| "拓扑相变 → 跨界顿悟" | FourDMem 计算 Betti 数 → 推送 `phase_transition` 信号 → Agent 用 LLM 做类比推理 |
| "范式转移 → 自我推翻" | FourDMem 监控同化失败率 → 推送 `paradigm_crisis` 信号 → Agent 用 LLM 做辩证综合 |
| "怪圈自指 → 修改自身规则" | FourDMem 检测效率低下 → 推送 `observer_alert` 信号 → Agent 决定是否调整参数 |
| "器官自生长 → Auto-Plugin" | FourDMem 检测检索痛点 → 推送 `plugin_needed` 信号 → Agent 用 LLM 生成插件代码 |
