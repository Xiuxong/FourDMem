# FourDMem 自动化完善升级方案 v2.0

> **生成日期**: 2026-06-21
> **基线**: `docs/automation-optimization-plan.md` (2026-06-16)
> **对标**: Mem0 / Letta (MemGPT) / Zep / Cognee / MemOS (2026 主流开源 Agent 记忆方案)
> **状态**: Phase 1 执行中

---

## 一、现状与竞品差距分析

### 1.1 主流方案核心特征对比

| 特征 | Mem0 | Letta (MemGPT) | Zep | Cognee | **FourDMem 现状** |
|:---|:---|:---|:---|:---|:---|
| **记忆提取** | 全自动（零人工） | Agent 自主调用函数 | 全自动 fact extraction | 自动 ingest pipeline | ⚠️ 依赖 Agent 调 log_turn，常忘 |
| **去重** | 向量相似度自动合并 | 语义去重 + 版本覆盖 | 实体解析 + 时序合并 | 图去重 | ❌ 无去重，重复事实堆积 |
| **分层** | user/session/agent 三 scope | in-context/archival 两层 | fact/entity/episode | L0-L3 知识图谱 | ✅ L0-L4 五层，设计更完整 |
| **时序感知** | 基本时间戳 | 有限 | ✅ temporal KG | 有限 | ✅ 主观时间 Active Ticks（设计领先） |
| **自动修剪** | 自动遗忘低价值 | 内存压力触发压缩 | 自动 | 自动 | ⚠️ DreamPruner 实现了但未运行 |
| **冷启动** | 无特殊处理 | 自动加载 archival | 自动 | 自动 | ⚠️ wake_up 只返回统计，无上下文恢复 |
| **知识图谱** | ✅ Graph Memory | ❌ | ✅ Temporal KG | ✅ | ⚠️ L1 图在 Rust 中，无跨层图关联 |
| **MCP 集成** | 有 | 有 | 有 | 无 | ✅ 有，但 Tool 不完整 |
| **后台自动化** | 全自动 | 全自动 | 全自动 | 全自动 | ❌ 进化引擎全部手动 |

### 1.2 FourDMem 独有优势（需保留并强化）

1. **四维正交坐标** `(embedding, neighbors, time_version, abstraction_level)` — 竞品无此设计
2. **主观时间 (Active Ticks)** — 解决长期停机断崖失忆，竞品均基于物理时间
3. **认知进化引擎** — 髓鞘化/拓扑相变/范式转移/怪圈自指，竞品无类似机制
4. **白盒可审计** — L0(SQLite) + L1(JSON) + L2(MD) + L3(YAML) 全人类可读
5. **RIF-U 评分** — 引入 Utility 反馈维度，比纯 RIF 更精准

### 1.3 核心差距（按严重程度排序）

| # | 差距 | 影响 | 竞品参考 |
|:---:|:---|:---|:---|
| **G1** | log_turn 靠 Agent 自觉调用 | 对话丢失率高 | Mem0/Zep 全自动提取 |
| **G2** | 认知进化引擎未自动化 | 核心卖点形同虚设 | — |
| **G3** | wake_up 无上下文恢复 | 冷启动体验差 | Letta 自动加载 archival |
| **G4** | 4 个 Tool 未注册 MCP | Agent 能力不完整 | — |
| **G5** | 无语义去重 | 重复事实堆积 | Mem0 向量去重 |
| **G6** | 无跨会话持久化 | 进程重启全丢 | 所有竞品均持久化 |
| **G7** | 无记忆健康自检 | 不知道记忆库状态 | Zep 有 health metrics |
| **G8** | Token 预算配额未实现 | 输出质量不可控 | — |

---

## 二、升级方案（6 大方向，18 项具体方案）

### 方向 A：全自动记忆采集（解决 G1）

> 对标 Mem0 / Zep 的"零人工提取"，消除对 Agent 调用 log_turn 的依赖。

#### A1. MCP 消息中间件自动捕获

**现状**：`_auto_archive()` 只在 Agent 调用 MCP Tool 时触发。非工具对话完全丢失。

**方案**：
- 利用 FastMCP 的 `@mcp.prompt()` 机制，在每次 prompt 注入时自动携带 log_turn 指令（软约束）
- 在 MCP Server 的 `main()` 中注册 signal handler，进程退出前自动 dump 未归档消息
- 如 MCP SDK 支持 request/response middleware，在 middleware 层拦截所有通信自动归档
- 验收：Agent 不调 log_turn 的情况下，仍有 ≥80% 对话被自动归档

#### A2. 语义触发的 L0→L1 提取

**现状**：`_post_interaction()` 每 10 次交互盲触发，不管对话质量。

**方案**：
- 新增 `SalienceDetector`：轻量关键词/模式匹配（决策词: must/should/architecture/bug/fix/prefer；中文: 必须/架构/决定/修复/偏好）
- 高价值对话立即触发提取，低价值跳过
- 提取后自动调 `resolve_conflicts()`
- 提取的 fact 自动打 `active_tick` 和 `shelf_life_category`
- 验收：高价值对话 ≤2 轮内完成 L1 提取

#### A3. 语义去重引擎

**现状**：`add_fact()` 直接写入，无去重。

**方案**：
- 写入前向量检索 Top-3，相似度 > 0.92 合并（更新 `last_active_tick` + 累加 frequency）
- 相似度 0.75-0.92 建立 `similar_to` 边（软关联）
- 相似度 < 0.75 正常写入
- 参考 Mem0 的 `add` 流程：先 search → 判断 add/update/noop
- 验收：连续写入 10 条相似事实，L1 图中只新增 ≤3 个节点

---

### 方向 B：认知进化引擎自动化（解决 G2）

> 将已实现但手动调用的进化组件接入自动化流水线。

#### B1. ObserverNode 常驻化

**现状**：`strange_loop.py` 的 `ObserverNode` 是纯 Python 类，无调用入口。

**方案**：
- 在 `_post_interaction()` 中每次 query 后自动调 `observer.observe(query_result)`
- 检测到 confidence crisis（连续 N 次置信度 < 0.3）时自动调参（via `hot_swap_dna`）
- 检测到 layer starvation 时自动调整配额
- 验收：构造低置信度场景，Observer 自动干预

#### B2. DreamPruner 后台化

**现状**：`DreamPruner.run_periodic()` 是同步死循环，未接入调度器。

**方案**：
- 改为 `threading.Thread` 后台任务，在 MCP Server `main()` 中启动
- 每次 `advance_tick()` 后检查 `tick % 100 == 0` 触发修剪
- 修剪报告写入 L0
- 验收：运行 500+ ticks 后低价值 L1 节点被清理

#### B3. 范式转移自动监控

**现状**：`ParadigmShiftEngine.check_crisis()` 需手动调用。

**方案**：
- `submit_feedback` 中 score < -0.5 时自动累加 failure count
- 超阈值自动触发辩证流程，结果写入 L2
- 验收：连续注入负面反馈，系统自动生成辩证 L2 规则

#### B4. 进化调度器统一入口

**方案**：
- 新增 `python/daemon/evolution_scheduler.py`
- 整合 ObserverNode + DreamPruner + ParadigmShiftEngine
- 暴露 `evolution_status()` MCP Tool
- 验收：`make run-mcp` 后进化引擎自动运行

---

### 方向 C：冷启动与上下文恢复（解决 G3）

> 对标 Letta 的 archival memory 自动加载。

#### C1. wake_up 上下文恢复

**现状**：返回 `{"l0_evidence": N, "l1_nodes": M}` — 对 Agent 无用。

**方案**：返回升级为：
```json
{
  "last_session_summary": "...",
  "recent_facts": ["fact1", "fact2"],
  "pending_decisions": [...],
  "memory_health": {...},
  "evolution_alerts": [...],
  "git_changes_since_last": "..."
}
```
- 冷启动时自动触发一次 L0→L1 提取
- 物理停机 > 7 天时标记旧记忆 `[NEEDS_VERIFICATION]`
- 验收：停机 1 天后重启，Agent 能说出上次讨论到哪

#### C2. 跨会话记忆持久化

**现状**：`get_engine()` 默认 `db_path=":memory:"`。

**方案**：默认值改为 `data/vault/evidence.db`，graph.json 自动持久化。
- 验收：kill 进程 → 重启 → wake_up 返回上次数据

#### C3. 知识断层反思 (Gap Reflection)

**方案**：启动时对比 `data/snapshots/last_session.json` 与当前 Git 状态，差异写入 L2。

---

### 方向 D：MCP Tool 完整性（解决 G4）

#### D1. 注册缺失 Tool

`save_memory`、`reflect`、`abandon_branch`、`get_entity_context` 加 `@mcp.tool()` 装饰器。

#### D2. 新增 memory_health Tool

返回活跃/过期/冲突/孤立节点统计 + 修剪建议 + 进化引擎状态。

#### D3. 新增 write_scenario Tool

让 Agent 能主动写入 L2 场景块，自动索引。

---

### 方向 E：工程化自动化（融合原方案 1-6, 8-10）

| 序号 | 方案 | 优先级 | 工时 |
|:---:|:---|:---:|:---:|
| E1 | 修复 MCP Server 启动（pyproject.toml 子包声明） | P0 | 0.5h |
| E2 | 启用 CI Python 测试（取消注释 ci.yml） | P0 | 1h |
| E3 | 增强 Makefile（dev/check-all/db-reset/mcp-test/evolution-smoke） | P1 | 2h |
| E4 | 跨层引用完整性检查 | P1 | 2h |
| E5 | Nightly + 依赖审计 | P2 | 1h |
| E6 | 性能基准 CI | P2 | 2h |
| E7 | MCP 健康监控 | P2 | 2h |
| E8 | Release 自动化 | P3 | 2h |
| E9 | 自动文档生成 | P3 | 1.5h |

---

### 方向 F：高级自动化（竞品差异化）

#### F1. 认知髓鞘化（思维宏编译）

高频成功路径（≥10 次，成功率 ≥90%）自动编译为思维宏，下次绕过检索直接返回。

#### F2. 拓扑相变监控

轻量版：定期计算 L1 图连通分量、平均度、聚类系数，临界时触发跨域类比搜索。

#### F3. 主观时间冷启动免疫

wake_up 时计算 tick 差值，差值 > 90 标记 `[HISTORICAL_CONTEXT]`，> 365 触发全面断层反思。

---

## 三、实施路线图

### Phase 1：基础可用（~5h）

| # | 方案 | 工时 | 状态 |
|:---:|:---|:---:|:---:|
| 1 | E1. 修复 MCP Server 启动 | 0.5h | ✅ |
| 2 | C2. 持久化存储默认值 | 0.5h | ✅ |
| 3 | D1. 注册缺失 MCP Tool | 1h | ✅ |
| 4 | C1. wake_up 上下文恢复 | 2h | ✅ |
| 5 | E2. 启用 CI Python 测试 | 1h | ✅ |

**里程碑**：MCP Server 可启动、持久化、Tool 完整、冷启动有上下文。

### Phase 2：自动记忆采集（~10.5h）

| # | 方案 | 工时 | 状态 |
|:---:|:---|:---:|:---:|
| 6 | A2. 语义触发 L0→L1 | 2h | ✅ |
| 7 | A3. 语义去重引擎 | 2h | ✅ |
| 8 | A1. MCP 消息中间件 | 3h | ✅ |
| 9 | D2. memory_health Tool | 1h | ✅ |
| 10 | D3. write_scenario Tool | 1.5h | ✅ |

**里程碑**：对话自动采集、去重、Agent 可写 L2。

### Phase 3：认知进化自动化（~10.5h）

| # | 方案 | 工时 | 状态 |
|:---:|:---|:---:|:---:|
| 11 | B1. ObserverNode 常驻化 | 2h | ✅ |
| 12 | B2. DreamPruner 后台化 | 1.5h | ✅ |
| 13 | B3. 范式转移自动监控 | 2h | ✅ |
| 14 | B4. 进化调度器统一入口 | 2h | ✅ |
| 15 | F1. 认知髓鞘化 | 3h | ✅ |

**里程碑**：进化引擎全自动运行。

### Phase 4：工程化完善（~8h）

| # | 方案 | 工时 | 状态 |
|:---:|:---|:---:|:---:|
| 16 | E3. 增强 Makefile | 2h | ✅ |
| 17 | E4. 跨层引用完整性 | 2h | ✅ |
| 18 | E5-E9. CI/CD 完善 | 4h | ✅ |

### Phase 5：差异化竞争（~9h）

| # | 方案 | 工时 | 状态 |
|:---:|:---|:---:|:---:|
| 19 | C3. 知识断层反思 | 3h | ✅ |
| 20 | F2. 拓扑相变监控 | 4h | ✅ |
| 21 | F3. 主观时间冷启动免疫 | 2h | ✅ |

---

## 四、与原方案差异

| 原方案 | 本次变化 | 原因 |
|:---|:---|:---|
| 方案 1-6, 8-10 | 保留，归入方向 E | 工程化基础不变 |
| 方案 7（进化调度器） | 升级为 B1-B4 | 拆解为 Observer/DreamPruner/ParadigmShift/统一调度 |
| 无 | 新增方向 A | 对标 Mem0/Zep 零人工提取，原方案最大盲区 |
| 无 | 新增方向 C | 对标 Letta archival 自动加载 |
| 无 | 新增方向 D | 原方案未提及 4 个未注册 Tool |
| 无 | 新增方向 F | 发挥四维架构独特优势 |
