# 🚀 Agent 永久记忆引擎 (Memory-Core) v3.0 开发任务清单

> **版本**: v3.0 (四维升维版 · 融合 v2.1 全部优化)
> **更新日期**: 2026-06-15
> **项目名称**: Agent Memory Core
> **目标**: 为 Claude Code、Cursor、Cline、OpenCode 等 AI 编程 Agent 构建高性能、白盒可控、具备四维认知结构的永久记忆基础设施
> **核心原则**: Rust 沉底 + Python 上浮 + MCP/A2A 双协议 + L0-L3 分层存储 + 时间版本树 + 仿生生命周期

---

## 📌 项目全局约束与原则 (v3.0 更新)

1.  **四维正交原则**: 任何一条记忆的存取必须同时具备四个坐标：`(embedding, graph_neighbors, time_version, abstraction_level)`。缺少任一维度的写入应被拒绝或自动补全。
2.  **L0-L3 分层存储 (Abstraction Hierarchy)**:
    -   **L0 (Raw Evidence)**: 原始对话/工具调用日志。不可变，只追加。是真相底座。
    -   **L1 (Atomic Facts)**: 结构化事实/指令。从 L0 提取，带 `source_l0_ids` 锚点。
    -   **L2 (Scenario Blocks)**: 场景化知识块。由多条 L1 聚合而成，Markdown 格式，可读性强。
    -   **L3 (Persona/Core)**: 稳定画像/核心规范。极少变动，全局共享，永不衰减。
3.  **时间作为独立版本轴 (Temporal Versioning)**: 不再仅用 `valid_from/to` 属性标记边，而是为每个实体维护一棵 **Version Tree**。查询时可指定 `at_time=T` 获取该时刻的精确快照；默认强制过滤过期版本，仅在显式历史查询时放开。
4.  **跨层索引与渐进披露 (Cross-Layer & Progressive Disclosure)**: 高层(L2/L3)必须持有指向低层(L1/L0)的可追溯索引。检索默认从高层开始（省 Token），需要证据时沿索引下钻（Drill-Down）。
5.  **Token 预算的四维配额分配**: 1500 Token 预算按**层级配额**分配：L3(20%) + L2(40%) + L1(30%) + L0(10%)。各层内部再按 RIF (Recency-Importance-Frequency) 分数排序填充。确保上下文既有宏观指导，又有微观证据。
6.  **白盒与明文优先 (White-Box First)**: L0 使用 SQLite+FTS5；L1 使用 `graph.json`；L2 使用 Markdown 文件；L3 使用 YAML/JSON。所有层级人类可读、可 Git 追踪。二进制缓存仅用于加速。
7.  **安全与沙箱**: MCP Server 对输入严格校验，支持多 Workspace 物理隔离，防止跨项目记忆泄漏与 Prompt 注入。

---

## 📦 Epic 1: 项目基础设施与工程化 (Phase 0)

**目标**: 搭建跨语言混合工程骨架，确保 CI/CD 和本地开发体验顺畅。

| 任务 ID | 任务名称 | 具体实现细节 | 推荐依赖/Crate | 验收标准 (DoD) |
| :--- | :--- | :--- | :--- | :--- |
| **T-1.1** | Cargo Workspace 与 Python 虚拟环境初始化 | 创建 Rust workspace，包含 `storage-core`, `graph-core`, `retrieval-core`, `sync-engine`, `py-bindings` 五个子 crate；配置 Python `pyproject.toml`。 | `maturin`, `uv` | 运行 `maturin develop` 可成功在 Python 中 `import` 空的 Rust 模块。 |
| **T-1.2** | 跨语言测试框架搭建 | 配置 Rust 单元测试与 Python 集成测试（pytest），实现代码覆盖率统计。 | `cargo-llvm-cov`, `pytest` | `make test` 可一键运行 Rust 和 Python 测试，并输出覆盖率报告。 |
| **T-1.3** | 代码规范与 Pre-commit Hooks | 配置 Rust 格式化、Clippy 静态检查，以及 Python 的 Ruff 检查。 | `rustfmt`, `clippy`, `ruff`, `pre-commit` | 提交代码时自动触发格式化和 lint 检查，不通过则拦截 commit。 |
| **T-1.4** | 目录结构与 Git 忽略规则定义 | 建立标准目录结构（见附录 A），严格配置 `.gitignore`，确保 `.cache/` 和 `.index/` 等二进制运行时文件不被提交。 | 无 | 目录结构清晰，`git status` 不会误报缓存文件。 |
| **T-1.5** | 开发者文档与架构知识库 | 编写四维架构设计文档、L0-L3 数据流图、跨语言调用序列图、新增 Tool 开发指南；配置 MkDocs 自动发布。 | `mkdocs`, `mermaid` | 新成员可按文档在 30 分钟内完成本地开发环境搭建。 |

---

## 🗄️ Epic 9: 四维存储架构 (L0-L3 分层栈)

**目标**: 废弃单一图结构，建立 L0-L3 物理分离 + 逻辑统一的四维存储栈。**本 Epic 为 v3.0 核心 Breaking Change**。

| 任务 ID | 任务名称 | 具体实现细节 | 推荐依赖/Crate | 验收标准 (DoD) |
| :--- | :--- | :--- | :--- | :--- |
| **T-9.1** | **L0 原始证据存储引擎** | 基于 SQLite + FTS5 实现。每条记录包含：`session_id`, `timestamp`, `raw_content`, `tool_calls`, `checksum`。**只写不删**。支持按 session/time 范围快速检索。 | `rusqlite`, `fts5` | 写入 1万条对话日志 < 2s；按时间范围查询 P99 < 5ms。 |
| **T-9.2** | **L1 原子事实图谱改造** | 改造现有 `graph-core`。节点增加 `layer=1` 标记和 `source_l0_refs: Vec<L0Id>` 字段。边保留时序属性，但改为指向 Version Tree 节点（见 Epic 10）。 | `petgraph`, `serde` | L1 节点能 100% 追溯到 L0 原始记录；图遍历性能不退化。 |
| **T-9.3** | **L2 场景块文档存储** | 新增 Markdown 文件存储区 (`vault/scenarios/`)。每个 L2 文档 Frontmatter 包含：`related_l1_ids`, `scenario_type`, `last_consolidated_at`。内容为人可读的结构化知识。 | `gray_matter`, `pulldown-cmark` | L2 文档能被 Tantivy 全文索引；修改 L2 文档能反向同步到 L1 关联关系。 |
| **T-9.4** | **L3 核心画像存储** | 独立 YAML/JSON 文件 (`vault/persona/`)。包含用户偏好、架构决策、编码规范等。加载时直接注入 System Prompt 或 Rule Daemon。**不参与常规检索排序**，始终置顶。 | `serde_yaml` | L3 内容在每次会话启动时 < 10ms 加载完毕；格式校验失败时降级为空画像而非崩溃。 |
| **T-9.5** | **跨层引用完整性校验器** | 后台 Daemon 定期扫描：L1→L0 引用是否断裂；L2→L1 引用是否失效；L3 引用的 L2 是否存在。发现孤儿节点自动标记或归档。 | 无 | 每日巡检报告无 CRITICAL 级引用断裂；修复操作可手动触发。 |

---

## ⏳ Epic 10: 时间版本树 (Temporal Version Tree)

**目标**: 将时间从“过滤属性”升级为“独立查询维度”，支持精确历史回溯与因果推理。**融合 v2.1 Temporal Gate 优化**。

| 任务 ID | 任务名称 | 具体实现细节 | 推荐依赖/Crate | 验收标准 (DoD) |
| :--- | :--- | :--- | :--- | :--- |
| **T-10.1** | **Version Tree 数据结构设计** | 为每个 L1 实体维护版本链：`{entity_id, version_seq, valid_range, content_hash, prev_version_ptr}`。使用 Rust `BTreeMap<Timestamp, Version>` 实现高效范围查询。 | `chrono`, `serde` | 单实体 1000 个版本下，查找 `at_time=T` 耗时 < 1μs。 |
| **T-10.2** | **写入时自动版本分叉** | `save_memory` 更新已有实体时，不覆盖旧值，而是创建新版本节点，旧版本 `valid_range.end = now`，新版本 `valid_range.start = now`。保留完整变更历史。 | 无 | 连续更新同一事实 10 次，能查到全部 10 个历史版本及精确时间戳。 |
| **T-10.3** | **[融合] 强制时序过滤器 (Temporal Gate)** | 在检索管线最前端插入 Temporal Gate。默认行为：所有 `valid_range.end < now()` 的版本**直接从候选集中剔除**。仅当查询参数显式传入 `include_expired=true` 或 `at_time` 时放开。Rust 侧实现，零性能开销。 | 无 | 默认模式下过期版本泄漏率 = 0%；显式历史查询可正确返回并附带 `[EXPIRED]` 标记。 |
| **T-10.4** | **时间点快照查询 API** | 新增 `query_at(query, timestamp)` 接口。Rust 侧将所有候选实体的版本树对齐到指定时间点，返回该时刻的有效状态集合。 | 无 | 查询"上周三的架构决策"能精确返回当时的版本，而非最新版本。 |
| **T-10.5** | **Diff 与变更原因追溯** | 新增 `get_diff(entity_id, v1, v2)` 工具。对比两个版本的 L1 内容差异，并自动关联导致变更的 L0 原始对话（通过 `source_l0_refs`）。 | 无 | Agent 问"这个配置为什么改了？"能返回 diff + 触发变更的原始聊天记录。 |
| **T-10.6** | **版本压缩与归档策略** | 对超过 90 天且未被访问的历史版本，合并为"摘要版本"（保留关键变更点，丢弃中间态）。原始 L0 证据永不压缩。 | 无 | 版本树深度控制在合理范围；压缩后仍能回答"大致何时发生了什么变化"。 |

---

## 🧠 Epic 2: Rust 核心图谱引擎 (graph-core)

**目标**: 构建高性能、线程安全的 L1 层图结构，**承载四维模型中的关联拓扑维度**。

| 任务 ID | 任务名称 | 具体实现细节 | 推荐依赖/Crate | 验收标准 (DoD) |
| :--- | :--- | :--- | :--- | :--- |
| **T-2.1** | **[升维] L1 图谱基础结构定义** | 使用 `petgraph::StableGraph`。定义 `NodeAttr` (ID, layer, source_l0_refs, importance_score, access_count, heat_score)；`EdgeAttr` 包含 `version_ptr` (指向 Version Tree), `backlinks` (反向锚点链接)。 | `petgraph`, `serde`, `chrono` | 能够创建带层级标记和版本指针的边；支持锚点反向链接。 |
| **T-2.2** | 并发控制与快照读模型 | 使用 `Arc<parking_lot::RwLock<Graph>>` 包装图结构。实现 `read_snapshot()` 方法，读操作获取 Arc clone，避免读写阻塞。 | `parking_lot` | 在 1 个写线程持续更新图谱时，10 个读线程并发查询无死锁、无崩溃。 |
| **T-2.3** | 可控多跳遍历与锚点展开 | 封装业务逻辑：支持最大跳数、关系类型白名单、权重阈值过滤、**锚点链追溯（沿 backlinks 展开原始碎片）**。新增 `expand_backlinks(node_id)` 方法，递归展开蒸馏节点的原始碎片。 | `petgraph::visit` | 1万节点规模下，2跳条件过滤遍历耗时 < 5ms；锚点展开能正确追溯到 L0 记录。 |
| **T-2.4** | 双格式持久化引擎 | 实现 `bincode` 高速二进制序列化（用于 `.cache/`）；实现 `serde_json` 导出/导入（用于人类可读和 Git 管理）。 | `bincode`, `serde_json`, `lz4_flex` | 1万节点图谱：Bincode 加载 < 10ms；JSON 导出格式兼容标准。 |
| **T-2.5** | **[降级为Phase2] 无阻塞写缓冲与双图切换** | *MVP阶段暂不实现，使用 RwLock 即可。Phase 2 引入 `arc-swap` 实现批量写入时的原子替换。* | `arc-swap` | (Phase 2 验收) 批量更新 500 个节点期间，并发检索 P99 延迟无毛刺。 |

---

## 🔍 Epic 3: 四维检索与重排引擎 (retrieval-core)

**目标**: 实现跨层、跨时间的智能检索，**融合 v2.1 RIF 评分并按层级配额分配 Token 预算**。

| 任务 ID | 任务名称 | 具体实现细节 | 推荐依赖/Crate | 验收标准 (DoD) |
| :--- | :--- | :--- | :--- | :--- |
| **T-3.1** | HNSW 向量索引集成 | 引入 HNSW 库。实现向量的增删改、持久化，支持增量更新。备选：`hnswlib-rs` 或 FFI 调用 C++ hnswlib。 | `usearch` / `hnswlib-rs` | 10万条 768维向量：Top-K 查询耗时 < 5ms，索引可序列化到磁盘。 |
| **T-3.2** | Tantivy 全文检索集成 | 建立 L2 MD 内容 + L0 FTS5 联合倒排索引。优化中文分词：使用 `cangjie` 或 IK 分词器插件。 | `tantivy`, `cangjie` | 10万文档规模：关键词查询耗时 < 10ms，中英文混合查询准确率高。 |
| **T-3.3** | RRF 融合重排器实现 | 实现 Reciprocal Rank Fusion 算法，接收图谱、向量、全文三路结果，根据权重计算最终排序得分。 | 无 (纯算法) | 给定三路排序列表，RRF 融合计算耗时 < 1ms。 |
| **T-3.4** | 索引健康度监控与自动重建 | 定期检测 HNSW 碎片率、Tantivy 段文件体积。碎片率 > 30% 时自动触发后台重建。 | 无 | 索引劣化后 10 分钟内自动重建完毕，服务无降级感知。 |
| **T-3.5** | **[融合] 分层 RIF 评分模型** | RIF 评分仅在**同层内**生效。L3 永远满分；L2 按热度+时效；L1 按完整 RIF 公式 (`α×recency + β×importance + γ×frequency`)；L0 按时间邻近度。跨层比较由配额机制决定，而非分数。重要性分级规则继承 v2.1。 | 无 (纯算法) | 同层内高 importance 记忆保留率显著高于低 importance；评分计算耗时 < 1ms。 |
| **T-3.6** | **[新增] Token 预算层级配额分配器** | 替代 v2.1 线性裁剪。实现配额管理器：L3(300t) + L2(600t) + L1(450t) + L0(150t)。各层内部按 RIF 排序填充。某层未用完配额可向下层转移，但不可向上侵占。 | `tiktoken` (Rust版) | 极端候选集下，输出始终包含高层指导和底层证据；Token 利用率 > 95%。 |
| **T-3.7** | **[新增] 分层检索路由器 (Layer Router)** | 根据 Query 意图自动选择检索起点：① 事实类 → L1/L2；② "为什么/怎么来的" → L0 + Diff；③ 偏好/规范类 → L3 直取；④ 历史状态类 → Version Tree。 | 无 | 路由准确率 > 90%；误路由可通过 `force_layer` 参数纠正。 |
| **T-3.8** | **[新增] 跨层一致性校验** | 检索结果返回前，检查 L2 描述与 L1 最新事实是否矛盾。若矛盾，优先返回 L1 最新事实 + 警告标记。 | 无 | 过时的高层摘要不会误导 Agent；矛盾情况有明确提示。 |

---

## ⚙️ Epic 4: 异步 IO 与自动化同步引擎 (sync-engine)

**目标**: 实现 L2 MD 文件、L3 YAML 与图谱/索引的自动化双向同步。

| 任务 ID | 任务名称 | 具体实现细节 | 推荐依赖/Crate | 验收标准 (DoD) |
| :--- | :--- | :--- | :--- | :--- |
| **T-4.1** | 异步批量文件解析器 | 基于 Tokio 并发读取 L2 MD 和 L3 YAML，解析 Frontmatter 和双链，提取实体与关系。 | `tokio`, `gray_matter`, `serde_yaml` | 并发解析 1000 个文件耗时 < 500ms。 |
| **T-4.2** | 文件监听与防抖机制 | 监听 vault 目录变更，引入防抖（Debounce）机制合并 500ms 内的连续事件。 | `notify`, `tokio::time` | 快速连续保存同一文件 5 次，仅触发 1 次更新任务。 |
| **T-4.3** | 增量同步与哈希校验兜底 | 计算文件内容 Hash (Blake3)，仅更新变更节点。增加定时全量 Hash 比对。 | `blake3`, `dashmap` | 修改单个文件增量更新耗时 < 50ms；定时巡检无遗漏。 |

---

## 🐍 Epic 5: PyO3 绑定与 Python 业务层 (py-bindings)

**目标**: 将 Rust 四维能力无缝暴露给 Python，**实现动态按需摘要与渐进式披露**。

| 任务 ID | 任务名称 | 具体实现细节 | 推荐依赖/Crate | 验收标准 (DoD) |
| :--- | :--- | :--- | :--- | :--- |
| **T-5.1** | PyO3 核心接口暴露 | 将 `StorageCore` (含 L0-L3) 封装为 Python 类。实现 `query()`, `query_at()`, `drill_down()`, `add_node()` 等。 | `pyo3`, `numpy` | Python 调用返回字典/列表，无内存泄漏，异常可被捕获。 |
| **T-5.2** | 惰性 NetworkX 视图转换 | 实现 `to_networkx(subgraph_query)`，仅将 Rust 返回的局部子图转换为 NetworkX 对象。 | `networkx` | 提取 50 个节点的子图并转换耗时 < 5ms。 |
| **T-5.3** | 全链路日志与可观测性 | Rust 层 `tracing` → PyO3 → Python `loguru`，实现跨语言链路追踪。 | `tracing`, `loguru` | 日志能清晰看到 Rust 各阶段耗时与 Python 处理耗时。 |
| **T-5.4** | **[融合] 四维查询聚合与配额裁剪** | Rust 侧实现 `query_with_quota(query, filters, quota_config, rif_weights)`。流程：Layer Router → Temporal Gate → 多路召回 → RRF → 分层 RIF → 配额填充 → 硬截断。 | `tiktoken` (Rust版) | Rust 返回紧凑 JSON，严格符合层级配额；高 importance 记忆在同层内优先保留。 |
| **T-5.5** | **[融合] 动态按需摘要引擎** | Python 层智能摘要决策：≤200t 保留原文；>500t 自动调用 LLM 生成一句话摘要 + `original_ref`。带缓存避免重复摘要。 | `litellm`, `tiktoken` | 3 条长篇记忆自动生成摘要后总 Token ≤ 1500；摘要准确率 > 85%。 |
| **T-5.6** | **[升维] 渐进式披露服务 (Progressive Disclosure)** | Python 层实现 `drill_down(memory_id, target_layer)`。默认返回 L2/L3 摘要 + L1 关键事实；Agent 追问时自动触发下钻，补充 L0 证据或展开 L2 全文。 | 无 | 首次响应紧凑精准；二次追问无缝展开细节，无需重复提问。 |
| **T-5.7** | **[融合] 锚点展开服务 (Backlink Expander)** | Python 层 `expand_memory(memory_id, depth)` 调用 Rust `expand_backlinks()`，返回完整溯源链（L3→L2→L1→L0）。 | 无 | 任意节点可一键展示完整四维溯源链。 |

---

## 🤖 Epic 6: Agent 集成层 (MCP Server & Rule Daemon)

**目标**: 让 Agent 能感知并操作四维记忆空间，**融合 v2.1 全部 MCP 优化**。

| 任务 ID | 任务名称 | 具体实现细节 | 推荐依赖/Crate | 验收标准 (DoD) |
| :--- | :--- | :--- | :--- | :--- |
| **T-6.1** | MCP Server 基础框架搭建 | Python MCP SDK，stdio/SSE 传输，JSON Schema 校验。 | `mcp`, `pydantic` | 在 Claude Code / Cursor 中成功注册；非法输入返回友好错误。 |
| **T-6.2** | **[升维] `search_memory` (四维检索)** | 支持 `layer="auto\|L0\|L1\|L2\|L3"` 和 `at_time="ISO8601\|latest"` 参数。默认 `auto+latest`。调用 T-5.4 聚合 + T-5.5 摘要。返回附带 `layer`, `version_seq`, `source_file`, `confidence`, `importance_score` 等元数据。 | `tiktoken` | 默认返回精准紧凑且全部有效的记忆，总 token < 1500；支持层级和时间维度精确控制。 |
| **T-6.3** | **[融合] `save_memory` 与精准去重** | 写入前向量去重（Cosine > 0.92 重复，0.85-0.92 相关）。更新时自动创建 Version Tree 新版本。写入时自动评估 `importance_score` 并分配到对应层级。 | 无 | 连续记录同一事实，系统建立版本链而非平行重复；新写入自动携带层级和重要性分级。 |
| **T-6.4** | **[新增] `drill_down(memory_id, target_layer)` Tool** | 从高层记忆下钻到低层证据。如从 L2 场景块下钻到 L0 原始对话。 | 无 | 支持渐进式探索，避免一次性塞入过多细节。 |
| **T-6.5** | **[新增] `get_history(entity_id, from?, to?)` Tool** | 获取实体的版本变更时间线，含 diff 摘要和变更原因（关联 L0）。 | 无 | Agent 能理解知识的演化过程。 |
| **T-6.6** | **[新增] `promote_to_l3(l2_id, rule_summary)` Tool** | 允许 Agent 主动提议将某个经验提升为核心规范（需审批流）。 | 无 | Agent 参与记忆体系共建。 |
| **T-6.7** | 核心 Tool: `get_entity_context` | 传入文件路径/函数名，返回关联文件和历史修改时间线（基于 Version Tree）。 | 无 | 获取该文件的"前世今生"依赖关系和修改动机。 |
| **T-6.8** | **[融合] 动态 Rule Daemon (四维注入)** | 监听 Git 分支/目录变化。注入内容按层级组织：`## Core Principles (L3)` → `## Current Context (L2)` → `## Recent Decisions (L1)`。动态区块仅更新 L2/L1，L3 保持稳定。泛化支持多种 Agent 规则文件。 | `watchdog`, `jinja2`, `gitpython` | 切换分支后 2 秒内规则文件自动更新；L3 区块稳定不变。 |
| **T-6.9** | MCP 安全性与权限控制 | 输入消毒、Tool 访问白名单、预留 OAuth/API Key 认证接口。 | `pydantic`, `pyyaml` | 超长/控制字符请求被拒；可通过配置禁用写操作。 |
| **T-6.10** | **[融合] 多 Workspace 记忆物理隔离** | 按 `workspace_id` 严格物理隔离 L0 SQLite、L1 缓存、L2/L3 文件、向量索引。 | 无 | 同时运行两个不同项目的 Agent，记忆查询与写入互不干扰。 |
| **T-6.11** | 记忆可信度与四维溯源可视化 | 提供 `trace_memory(id)` 工具，展示完整 L3→L2→L1→L0 溯源链。 | 无 | Agent 可获取某条记忆的完整四维溯源路径。 |

---

## 🧬 Epic 8: 四维生命周期与自动化

**目标**: 蒸馏 = 自底向上的层级跃迁，**融合 v2.1 锚点机制与分级策略**。

| 任务 ID | 任务名称 | 具体实现细节 | 推荐依赖/Crate | 验收标准 (DoD) |
| :--- | :--- | :--- | :--- | :--- |
| **T-8.1** | **[融合] Auto-Capture Hook (L0→L1)** | 监听会话结束事件。对话日志写入 L0 后，异步触发 LLM 提取结构化事实写入 L1，建立 `source_l0_refs`。支持 `confirm_before_save`。 | `litellm` / Ollama | 对话结束后自动提取 ≥3 条高置信度 L1 记忆；L1 必有 L0 来源。 |
| **T-8.2** | **[升维] L1→L2 场景聚合器** | 替代 v2.1 扁平蒸馏。当某主题下 L1 碎片 ≥5 条时，LLM 生成 L2 场景文档，关联所有相关 L1。**保留所有 L1 原文**，L2 仅是"阅读视图"。锚点机制继承 v2.1。 | `scikit-learn`, LLM API | 生成的 L2 人可读、结构清晰；删除 L2 不影响 L1 完整性；所有 L2→L1 锚点有效。 |
| **T-8.3** | **[升维] L2→L3 核心规范晋升** | 当某 L2 被引用 ≥20 次且跨度 ≥30 天，系统提议晋升为 L3。需人工确认或 Agent 显式批准。晋升后原 L2 保留为历史参考。 | 无 | 核心规范沉淀路径清晰可审计；L3 更新频率极低但质量极高。 |
| **T-8.4** | **[融合] 四维遗忘策略** | L0: 永不删除，仅压缩存储；L1: 低置信度+长期未访问 → 归档；L2: 被新 L2 取代 → 旧版进入 Version Tree；L3: 仅手动删除或显式覆盖。衰减豁免继承 v2.1 (`importance ≥ 0.8` 不衰减)。 | 无 | 遗忘行为符合认知规律；关键证据永不丢失。 |
| **T-8.5** | **[新增] 蒸馏锚点升级 (Provenance Chain v2)** | v2.1 `backlinks` 升级为 `provenance_chain`: `{l3_rule → l2_scenario → [l1_facts] → [l0_evidence]}`。支持正向追溯和反向验证。 | 无 | 任意节点可一键展示完整四维溯源链。 |
| **T-8.6** | **[融合] A2A 记忆共享协议预留** | 基于 Google A2A 协议，允许主 Agent 将特定层级记忆授权给子 Agent 读取。 | `a2a` (协议草案) | 提供 API 接口，允许外部 Agent 通过 Token 授权拉取指定 Workspace 的只读记忆子集。 |

---

## 📊 Epic 7: 端到端集成、测试与调优

**目标**: 验证四维系统稳定性，**融合 v2.1 全部专项测试 + 新增四维专项测试**。

| 任务 ID | 任务名称 | 具体实现细节 | 推荐依赖/Crate | 验收标准 (DoD) |
| :--- | :--- | :--- | :--- | :--- |
| **T-7.1** | 基准性能测试 (Benchmark) | 1万节点/5万关系/10万向量。对比纯 Python、LightRAG 与本方案。**新增 RIF、Temporal Gate、Layer Router、配额分配的性能开销基准**。 | `criterion`, `pytest-benchmark` | 聚合查询 P99 < 150ms；四维附加开销总计 < 10ms。 |
| **T-7.2** | 内存泄漏与并发压测 | 50 并发 MCP 查询 + 持续后台写入 + Auto-Capture，运行 1 小时。 | `flamegraph`, `locust` | 持续压测 1 小时，内存无异常增长，Rust 层无 Panic。 |
| **T-7.3** | 白盒审计与灾难恢复演练 | 手动修改 JSON/YAML/MD、删除缓存、触发索引劣化，验证自愈和热加载。 | 无 | 删除缓存后重启自动重建；手动修改文件后系统正确热加载。 |
| **T-7.4** | Agent 真实场景验收测试 | 在真实 Claude Code / Cursor 中评估召回准确率、Token 节省率、Auto-Capture 准确率、四维溯源有效性。 | 无 | Agent 生成代码符合注入规范；Auto-Capture 准确提取核心决策。 |
| **T-7.5** | **[融合] 蒸馏信息保真度专项测试** | 1000+ 轮对话 → 多轮蒸馏。验证：① 所有 L2→L1 锚点完整；② 通过 `expand_backlinks` 可 100% 追溯到 L0；③ 热碎片未被误合并；④ 蒸馏后召回率 ≥ 蒸馏前。 | 无 | 蒸馏后信息零丢失，所有锚点有效，热碎片完好。 |
| **T-7.6** | **[融合] 时序过滤有效性专项测试** | 大量过期版本数据集。验证：① 默认搜索 0 条过期；② `at_time` 可正确返回历史版本；③ Agent 不会基于过时记忆生成错误代码。 | 无 | 过期版本泄漏率 = 0%；时间旅行查询 100% 准确。 |
| **T-7.7** | **[融合] Token 预算极端压力测试** | 100+ 条候选（混合层级/重要性/长度）。验证：① 层级配额合规 ±5%；② 高 importance 保留率 > 95%；③ 长文本正确摘要；④ 输出 ≤ 1500 Tokens。 | 无 | 极端候选集下输出精准紧凑，Token 预算零溢出。 |
| **T-7.8** | **[新增] 层级完整性压力测试** | 注入 10万条 L0，自动生成 L1/L2。验证：所有 L1 有 L0 来源；所有 L2 有 ≥3 条 L1 关联；无孤儿节点。 | 无 | 跨层引用完整性 100%；巡检报告无 CRITICAL 级问题。 |
| **T-7.9** | **[新增] 渐进式披露体验测试** | Agent 完成复杂调试任务。验证：首次检索获得足够上下文；追问时无缝获取底层证据；全程无信息过载或不足。 | 无 | 首次响应 Token 占用 < 800；追问后补充信息精准匹配需求。 |
| **T-7.10** | **[新增] v2.1 → v3.0 迁移工具验证** | 现有 `graph.json` + MD 文件自动迁移至 L0-L3 结构。验证迁移后检索结果语义等价。 | 无 | 迁移过程只读可用，写入暂停 ≤ 5 分钟；迁移后回归测试通过率 100%。 |

---

## 💡 附录

### 附录 A: 推荐的项目目录结构 (v3.0)

```text
agent-memory-core/
├── crates/                    # Rust 核心代码
│   ├── storage-core/          # [新增] L0 SQLite + L3 YAML 存储引擎
│   ├── graph-core/            # L1 图结构、Version Tree、锚点反向链接
│   ├── retrieval-core/        # 向量、全文、RRF、分层RIF、Temporal Gate、配额分配
│   ├── sync-engine/           # 文件监听、MD/YAML解析、增量同步
│   └── py-bindings/           # PyO3 封装层
├── python/                    # Python 业务与集成代码
│   ├── memory_core/           # PyO3 导入与业务逻辑封装
│   ├── mcp_server/            # MCP Server 实现与 Tool 定义
│   ├── summarizer/            # 动态按需摘要引擎 & 缓存
│   ├── daemon/                # Rule Daemon & 生命周期(做梦/衰减/晋升)
│   └── tests/                 # Pytest 集成测试
├── docs/                      # 四维架构文档与开发指南
├── data/                      # 数据目录 (用户项目级，受 Git 管理)
│   ├── vault/
│   │   ├── persona/           # L3 核心画像 (YAML/JSON)
│   │   ├── scenarios/         # L2 场景块 (Markdown)
│   │   └── evidence.db        # L0 原始证据 (SQLite)
│   ├── graph.json             # L1 人类可读图谱 (真相源)
│   └── .archive/              # 衰减归档区
├── .cache/                    # 运行时缓存 (Git 忽略)
│   ├── workspaces/            # 按 workspace_id 隔离
│   │   ├── ws_default/
│   │   └── ws_project_a/
│   └── ...
├── Cargo.toml
├── pyproject.toml
├── CLAUDE.md                  # 动态生成的 Agent 规则文件 (四维注入)
└── Makefile
