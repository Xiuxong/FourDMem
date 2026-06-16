# 🧠 Agent Memory Core v4.0

**认知进化与怪圈引擎 — 让 AI Agent 从“立体记忆”走向“硅基生命”**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](crates/)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](python/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-purple.svg)](python/mcp_server/)
[![Version](https://img.shields.io/badge/Version-4.0%20(Evolution)-ff69b4.svg)](#)

> ⚠️ **当前状态**: v4.0 架构设计与方案验证阶段。本版本在 v3.0 四维记忆基础上，引入了**认知进化与怪圈自指机制**，欢迎参与架构评审与 Issue 讨论。

## 🎯 项目愿景

当前的 AI 编程 Agent 普遍面临**记忆扁平化**与**思维固化**的双重困境：
1. **记忆扁平**：所有知识被压缩在同一层级的向量库中，缺乏抽象层次、时间纵深和证据溯源（v2.1 痛点）。
2. **思维固化**：即使拥有了完美的记忆库，Agent 的“思考方式”（检索策略、本体论、推理规则）依然是人类硬编码的。它只是一个“不断变大的立体硬盘”，无法像人类一样在经验积累中**产生直觉、顿悟跨界规律、甚至推翻旧有认知**（v3.0 局限）。

**Agent Memory Core v4.0** 提出**生命跃迁架构**，将记忆体从“被动的认知超立方体”升级为“主动进化的硅基大脑”。它不仅解决“记什么”，更解决“如何思考”，让 Agent 具备**认知髓鞘化（直觉）**、**拓扑相变（顿悟）**、**范式转移（自我推翻）** 与**怪圈自指（修改自身规则）** 的能力。

---

## 🏗️ 核心架构：L0-L4 五层栈 + 主观时间 + 进化引擎

| 层级 | 名称 | 存储格式 | 特性 | 生物学/认知学类比 |
|:---:|:---|:---|:---|:---|
| **L4** | **Meta-Cognition (Observer)** | **YAML/Python** | **[V4.0 新增]** 存储“关于如何思考的规则”。可被怪圈引擎自我指涉并动态修改。 | **前额叶皮层 / 意识** |
| **L3** | Persona / Core Rules | YAML/JSON | 核心规范。极少变动，但可被“范式转移引擎”在认知危机时重写。 | 价值观 / 核心信仰 |
| **L2** | Scenario Blocks | Markdown | 场景化知识块，人可读。支持条件化规则与跨域类比。 | 经验与叙事记忆 |
| **L1** | Atomic Facts | graph.json | 结构化事实，带版本树、主观时间刻度与效用锚点。 | 陈述性记忆 / 神经突触 |
| **L0** | Raw Evidence | SQLite+FTS5 | 原始对话日志，只追加不删除，绝对真相底座。 | 感觉记忆 / 海马体输入 |

### 🔑 六大设计原则

1.  **四维正交**: 每条记忆必须同时具备 `(embedding, neighbors, time_version, abstraction_level)` 四个坐标。
2.  **主观时间轴 (Subjective Time)**: 废弃纯物理时间衰减。引入 **`Active Ticks` (交互刻度)**，彻底解决 Agent 长期停机重启时的“断崖式失忆症”，支持冷启动环境唤醒。
3.  **渐进式披露**: 检索默认从高层(L2/L3)开始节省 Token，结合元认知置信度评估，追问时自动沿锚点下钻至 L0 证据。
4.  **Token 配额制**: 1500 Token 预算按 `L3(20%) + L2(40%) + L1(30%) + L0(10%)` 分层分配，各层内部按 **RIF-U (含效用反馈)** 排序。
5.  **白盒与明文优先**: L0(SQLite) + L1(JSON) + L2(MD) + L3/L4(YAML)。全部人类可读、可 Git 追踪，二进制缓存可随时重建。
6.  **认知进化与怪圈自指 (Strange Loop)**: 记忆体不仅是存储，更是思维进化的培养皿。系统必须具备自我重构拓扑、进化检索策略、甚至重写自身本体论的能力。

---

## 🧬 V4.0 核心特性：认知进化与怪圈引擎 (Cognitive Evolution Engine)

V4.0 跨越了传统软件工程的边界，从 5 个学科汲取灵感，构建了让 Agent “活着并成长”的底层机制：

| 学科视角 | 机制名称 | 工程实现 (How it works) | 涌现能力 |
|:---|:---|:---|:---|
| **🧠 生物学** | **认知髓鞘化 (Myelination)** | 监控推理路径调用频率。高频成功路径自动**编译为“思维宏 (Cognitive Macro)”**。 | **直觉反应**：遇到熟悉问题绕过繁琐检索，延迟下降 90%。 |
| **🌌 物理学** | **拓扑相变 (Phase Transition)** | 引入**拓扑数据分析 (TDA)** 监控 L1 图谱 Betti 数。达到自组织临界点时触发“认知雪崩”，利用 GNN 寻找高维同构流形。 | **跨界顿悟**：自动涌现跨学科类比（如领悟前端事件循环与后端并发的同构性）。 |
| **🏛️ 哲学** | **范式转移 (Paradigm Shift)** | 监控“同化失败率”。当旧经验连续导致 Bug，冻结写入，启动辩证 Agent 将“旧 L3(正题)”与“新 L0(反题)”辩论，生成“新 L3(合题)”。 | **自我推翻**：自主推翻过去的“核心教条”，完成认知升级。 |
| **♾️ 逻辑学** | **怪圈自指 (Strange Loop)** | 建立 **L4 观察者节点 (Observer)**。当系统陷入死锁或低效时，观察者跳出当前层级，**动态修改自身的 RIF-U 权重参数或 Prompt 模板**。 | **元认知觉醒**：意识到“我现在的思考方式有问题”，并改变思考方式。 |
| **🧬 复杂系统** | **器官自生长 (Auto-Plugin)** | 当发现某类问题（如复杂正则）检索极差时，LLM **自动编写专用 Python 检索插件**，沙盒验证后热插拔到 Retrieval Router。 | **开放式进化**：自己给自己“长出”新的大脑皮层区域。 |

---

## ⚡ 技术栈

| 层 | 技术选型 | 职责 |
|:---|:---|:---|
| **存储与图引擎** | Rust (`petgraph`, `rusqlite`, `tantivy`, `usearch`) | L0-L3 存储、Version Tree、图遍历、向量/全文索引、RIF-U 评分、配额裁剪 |
| **拓扑与进化计算** | Rust/Python (`scikit-tda`, `ripser`, `PyG`, `deap`) | **[V4.0]** TDA 持续同调计算、GNN 跨域流形寻找、认知 DNA 遗传算法 |
| **业务与认知逻辑** | Python (`pyo3`, `litellm`, `langchain`, `tiktoken`) | PyO3 绑定、动态摘要、反思 Agent、范式转移辩论、Auto-Plugin 生成与沙盒测试 |
| **Agent 集成** | MCP SDK + Rule Daemon + Observer | `search_memory` / `submit_feedback` / `abandon_branch` / `wake_up` 等 Tool |
| **数据真相源** | MD + JSON + YAML + SQLite + Python | 全部人类可读、可 Git 追踪，包含 Agent 自动生成的思维插件代码 |

---

## 📊 版本演进路线：从扁平到生命

| 维度 | v2.1 (扁平图谱) | v3.0 (四维超立方体) | **v4.0 (生命跃迁版)** |
|:---|:---|:---|:---|
| **核心定位** | 带标签的平面知识库 | 具备时间纵深与反思的立体记忆 | **具备自我进化能力的硅基大脑** |
| **存储结构** | 单一 `graph.json` + MD | L0-L3 物理分离 + 跨层索引 | **L0-L4 (含元认知观察者) + 认知基因组** |
| **时间处理** | `valid_from/to` 物理过滤 | 独立 Version Tree + Temporal Gate | **主观时间 (Active Ticks) + 休眠唤醒 + 断层反思** |
| **认知模式** | 被动检索，线性拼接 | 冲突反思，条件化记忆，元认知下钻 | **直觉编译、拓扑顿悟、范式转移、怪圈自指** |
| **系统边界** | 静态代码，人类维护 | 静态规则，人类调参 | **开放式进化，Agent 自主编写插件与修改自身规则** |
| **Token 分配** | RIF 分数线性裁剪 | 层级配额 + 同层 RIF 排序 | **层级配额 + RIF-U (含效用反馈) + 思维宏短路** |

---

## 🚀 快速开始 (概念验证)

```bash
# 1. 克隆项目
git clone https://github.com/your-org/agent-memory-core.git
cd agent-memory-core

# 2. 初始化 Rust 与 Python 混合环境
make init

# 3. 启动 MCP Server (包含冷启动唤醒与进化引擎守护进程)
make run-mcp

# 4. 运行认知进化专项测试 (验证范式转移与怪圈自指)
make test-evolution
