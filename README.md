# FourDMem
四维认知记忆引擎 | L0-L3 分层存储 + 时间版本树 | 为 AI 编程 Agent 构建高性能、白盒可控、具备仿生生命周期的永久记忆基础设施 | Rust + Python + MCP

# 🧠 Agent Memory Core v3.0

**四维认知记忆引擎 — 让 AI Agent 拥有真正的"立体记忆"**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](crates/)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](python/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-purple.svg)](python/mcp_server/)

> ⚠️ **当前状态**: v3.0 方案验证讨论阶段，尚未进入全面实施。欢迎参与架构评审与 Issue 讨论。

## 🎯 项目愿景

当前的 AI 编程 Agent（Claude Code、Cursor、Cline 等）普遍面临**记忆扁平化**困境：所有知识被压缩在同一层级的向量库或图谱中，缺乏抽象层次、时间纵深和证据溯源能力。这导致 Agent 在复杂项目中频繁出现"遗忘关键决策"、"引用过时信息"、"无法解释推理依据"等问题。

**Agent Memory Core v3.0** 提出**四维记忆模型**，将记忆从"带标签的平面图谱"升维为"认知超立方体"：

D1 语义空间 × D2 关联拓扑 × D3 时间版本 × D4 抽象层级
## 🏗️ 核心架构：L0-L3 四层栈 + 时间版本树

| 层级 | 名称 | 存储格式 | 特性 | 类比 |
|:---:|:---|:---|:---|:---|
| **L3** | Persona / Core Rules | YAML/JSON | 永不衰减，始终置顶，全局共享 | 人格与价值观 |
| **L2** | Scenario Blocks | Markdown | 场景化知识块，人可读，由 L1 聚合 | 经验与叙事记忆 |
| **L1** | Atomic Facts | graph.json | 结构化事实，带版本树与锚点 | 陈述性记忆 |
| **L0** | Raw Evidence | SQLite+FTS5 | 原始对话日志，只追加不删除 | 感觉记忆 / 证据底座 |

### 🔑 四大设计原则

1.  **四维正交**: 每条记忆必须同时具备 `(embedding, neighbors, time_version, abstraction_level)` 四个坐标
2.  **时间即坐标轴**: Version Tree 替代 `valid_from/to` 属性过滤，支持任意时间点精确快照与因果回溯
3.  **渐进式披露**: 检索默认从高层(L2/L3)开始节省 Token，追问时沿锚点下钻至 L0 证据
4.  **Token 配额制**: 1500 Token 预算按 `L3(20%) + L2(40%) + L1(30%) + L0(10%)` 分层分配，而非线性裁剪

## ⚡ 技术栈

| 层 | 技术选型 | 职责 |
|:---|:---|:---|
| **存储引擎** | Rust (`petgraph`, `rusqlite`, `tantivy`, `usearch`) | L0-L3 存储、Version Tree、图遍历、向量/全文索引、RIF 评分、配额裁剪 |
| **业务逻辑** | Python (`pyo3`, `litellm`, `tiktoken`) | PyO3 绑定、动态摘要、渐进式披露、生命周期管理 |
| **Agent 集成** | MCP SDK + Rule Daemon | `search_memory` / `drill_down` / `get_history` / `promote_to_l3` 等 Tool |
| **数据真相源** | MD + JSON + YAML + SQLite | 全部人类可读、可 Git 追踪，二进制缓存可随时重建 |

## 📊 v2.1 → v3.0 关键升级

| 维度 | v2.1 (扁平图谱) | v3.0 (四维超立方体) |
|:---|:---|:---|
| 存储结构 | 单一 `graph.json` + MD | L0-L3 物理分离 + 跨层索引 |
| 时间处理 | `valid_from/to` 过滤属性 | 独立 Version Tree + Temporal Gate |
| 抽象层级 | 所有节点混在同一层 | 严格 L0→L3 分层 + 自底向上跃迁 |
| Token 分配 | RIF 分数线性裁剪 | 层级配额 + 同层 RIF 排序 |
| 蒸馏机制 | 扁平合并 + 锚点反向链接 | L1→L2 场景聚合 + L2→L3 晋升 + Provenance Chain |
| 检索模式 | 一次性返回所有相关片段 | 渐进式披露 + Drill-Down |
