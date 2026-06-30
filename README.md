# 🧠 FourDMem

**四维认知记忆引擎 — L0-L4 五层架构 + 主观时间 + 认知进化**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](crates/)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](python/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-purple.svg)](python/mcp_server/)
[![CI](https://github.com/Xiuxong/FourDMem/actions/workflows/ci.yml/badge.svg)](https://github.com/Xiuxong/FourDMem/actions/workflows/ci.yml)

> 为 AI Agent 提供具备**记忆分层、语义去重、主观时间、认知进化**的长期记忆系统。通过 MCP 协议集成，支持 Claude / Cursor / 任意 MCP 客户端。

---

## ✨ 核心特性

- **L0-L4 五层记忆架构**: 原始证据 → 原子事实 → 场景知识 → 核心画像 → 元认知
- **四路召回 + RRF 融合**: Tantivy BM25 + SQLite FTS5 + 图遍历 + 向量 ANN
- **RIF-U 四维评分**: Recency + Importance + Frequency + Utility 反馈闭环
- **语义去重**: cosine > 0.92 自动合并，0.75-0.92 软链接
- **主观时间 (Active Ticks)**: 基于交互次数衰减，解决长期停机断崖失忆
- **认知进化引擎**: 髓鞘化(直觉) / 拓扑相变(顿悟) / 范式转移(自我推翻)
- **1500 Token 预算**: L3(20%) + L2(40%) + L1(30%) + L0(10%) 分层配额

---

## 🏗️ 架构总览

```text
Agent Query
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ L4  观察者层 ─ 元认知监控 & 认知进化                    │
│     ObserverNode / ParadigmShift / Myelination / TDA    │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│ L3  核心画像 ─ YAML persona (300 tokens, 20%)           │
│     core_rules / preferences / constraints              │
└─────────────────────────┬───────────────────────────────┘
                          ▲ AutoPromoter (≥20引用, ≥50 ticks)
┌─────────────────────────────────────────────────────────┐
│ L2  场景知识 ─ Markdown+YAML (600 tokens, 40%)          │
│     ScenarioBlock / 双通道索引 (FAISS + TraceMem)       │
└─────────────────────────┬───────────────────────────────┘
                          ▲ AutoAggregator (≥5条同主题L1)
┌─────────────────────────────────────────────────────────┐
│ L1  原子事实 ─ 知识图谱 (450 tokens, 30%)              │
│     petgraph + HNSW向量 + VersionTree + 语义去重        │
└─────────────────────────┬───────────────────────────────┘
                          ▲ FactExtractor (Agent LLM提取)
┌─────────────────────────────────────────────────────────┐
│ L0  原始证据 ─ SQLite+FTS5 (150 tokens, 10%)           │
│     append-only / jieba分词 / 停用词过滤                │
└─────────────────────────────────────────────────────────┘
```

---

## ⚡ 技术栈

| 层 | 技术 | 职责 |
|:---|:---|:---|
| **存储引擎** | Rust (`petgraph`, `rusqlite`, `tantivy`, `usearch`) | L0-L1 存储、图遍历、向量/全文索引、RIF-U 评分 |
| **认知逻辑** | Python (`pyo3`, `scikit-learn`, `networkx`) | 去重、提取、聚合、晋升、进化引擎 |
| **Agent 集成** | MCP SDK | 15 个 MCP Tool (search/feedback/extract/evolve) |

---

## 🚀 快速开始

### 前置条件

- Rust 1.75+
- Python 3.11+
- Git

### 安装

```bash
# 克隆项目
git clone https://github.com/Xiuxong/FourDMem.git
cd FourDMem

# 初始化 (安装 Rust 依赖 + Python venv + maturin 绑定)
make init

# 运行测试
make test
```

### 作为 MCP Server 使用

```bash
# 启动 MCP Server
make run-mcp

# 或直接运行
cd python && python -m mcp_server.server
```

在 Claude Desktop / Cursor 中配置 `.mcp.json`:

```json
{
  "mcpServers": {
    "FourDMem": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "python"
    }
  }
}
```

---

## 📊 Benchmark

```
Rigorous Recall (18q × 20f, eval=cosine_similarity(threshold=0.65), top_k=10)
------------------------------------------------------------------------------------------
Config                   Acc R@1 R@2 R@3 R@4 R@5 R@6 R@7 R@8 R@9 R@10      P50
------------------------------------------------------------------------------------------
text_only              100.0%  94% 100% 100% 100% 100% 100% 100% 100% 100% 100%   261.1ms
vector_only            100.0% 100% 100% 100% 100% 100% 100% 100% 100% 100% 100%   260.4ms
rrf_fusion             100.0% 100% 100% 100% 100% 100% 100% 100% 100% 100% 100%   260.8ms
full_pipeline          100.0% 100% 100% 100% 100% 100% 100% 100% 100% 100% 100%   260.7ms
------------------------------------------------------------------------------------------
```

---

## 🔧 MCP Tools

| Tool | 说明 |
|:---|:---|
| `search_memory` | 语义搜索记忆，自动唤醒 |
| `submit_feedback` | 对记忆有用性打分，更新 Utility |
| `extract_deep` | 提取原子事实到 L1 图谱 |
| `wake_up` | 冷启动恢复上下文 |
| `save_memory` | 显式保存关键事实 |
| `checkpoint_memory` | 持久化 L1 到磁盘 |
| `load_memory` | 从磁盘加载 L1 |
| `reflect` | 评估检索置信度 |
| `abandon_branch` | 标记废弃决策分支 |
| `get_entity_context` | 查看记忆的完整上下文 |
| `memory_health` | 记忆系统健康检查 |
| `write_scenario` | 写入 L2 场景块 |
| `synthesize_l2` | 综合多条记忆为场景 |
| `reflect_and_synthesize` | 辩证式知识综合 |
| `check_cognition_signals` | 检查认知信号 |

---

## 📁 项目结构

```
FourDMem/
├── crates/                    # Rust 核心引擎
│   ├── storage-core/          #   L0 SQLite + L2/L3 存储
│   ├── graph-core/            #   L1 图结构 + 版本树 + TDA
│   ├── retrieval-core/        #   全文/向量检索 + RRF + RIF-U
│   ├── evolution-core/        #   认知宏缓存 + 沙盒
│   ├── sync-engine/           #   文件同步 + 场景解析
│   └── py-bindings/           #   PyO3 Python 绑定
├── python/                    # Python 认知层
│   ├── mcp_server/            #   MCP Server + 生命周期
│   ├── cognition/             #   去重/提取/聚合/信号/治理
│   ├── evolution/             #   观察者/范式/髓鞘化/拓扑
│   ├── daemon/                #   进化调度器/健康监控
│   └── tests/                 #   98 个集成测试
├── benchmarks/                # 召回精度 benchmark
├── docs/                      # 架构文档
├── Makefile                   # 构建/测试/运行命令
└── Cargo.toml                 # Rust workspace
```

---

## 🧪 测试

```bash
make test           # 全部测试 (Rust + Python)
make test-rust      # Rust 单元测试 (143 tests)
make test-python    # Python 集成测试 (98 tests)
make bench          # Rust benchmark
make check-all      # fmt + lint + test + bench
```

---

## 📖 文档

- [FDRFlow.md](FDRFlow.md) — 检索全流程详解
- [RIF-SQCE.md](RIF-SQCE.md) — RIF-U 评分与配置参数
- [CHANGELOG.md](CHANGELOG.md) — 版本变更记录

---

## 🤝 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 📄 许可证

[MIT License](LICENSE)
