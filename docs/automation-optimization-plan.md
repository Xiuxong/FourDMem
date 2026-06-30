# 🚀 FourDMem 项目自动化优化方案

> **生成日期**: 2026-06-16
> **状态**: 待实施
> **范围**: CI/CD、构建系统、认知进化调度、运维监控、文档自动化

---

## 一、现状诊断

### 已具备的自动化

| 维度 | 当前状态 | 评价 |
|------|---------|------|
| **CI/CD** | `ci.yml` — Linux + Windows 双平台 Rust 测试，Python 部分**全注释** | ⚠️ 只有一半 |
| **代码规范** | `pre-commit-config.yaml` — rustfmt + clippy + ruff + trailing-whitespace | ✅ 基本完整 |
| **Makefile** | init / test / fmt / lint / clean / run-mcp | ✅ 可用但偏薄 |
| **MCP Server** | 已实现 `search_memory` / `submit_feedback` / `wake_up` / `log_turn` | ✅ 核心可用 |
| **自动归档** | `_auto_archive()` 每次交互自动 L0 写入 + 每 10 次 L0→L1 提取 | ✅ 有但粗糙 |
| **Rule Daemon** | 监听 L3/L2 文件变化并重编译规则 | ✅ 基础可用 |
| **认知进化** | `paradigm_shift.py` / `strange_loop.py` / `auto_plugin.py` / `extractor.py` | ✅ 框架在 |

### 关键缺陷

1. **CI Python 测试完全注释** — Python 代码零自动化验证
2. **无 nightly/定时构建** — 无法捕获依赖更新导致的回归
3. **无 Release/发布自动化** — 版本号手动维护
4. **MCP Server 自身无法启动** — 模块路径错误 (`ModuleNotFoundError: No module named 'mcp_server'`)
5. **Makefile 无增量构建** — 每次全量 `cargo check` + `maturin develop`
6. **无依赖安全扫描** — Cargo + pip 无 audit 流程
7. **无基准测试 CI** — `criterion` 基准只在本地，不追踪性能回归
8. **认知进化引擎无自动化调度** — `dream_pruning` / `myelination` 只有配置文件，无 Cron/Daemon

---

## 二、具体优化方案（10 项）

### 方案 1：修复 MCP Server 启动失败

**问题**：`.mcp.json` 中 `cwd` 设为 `python/`，但 `mcp_server` 子目录未安装为包，导致 `ModuleNotFoundError`。

**方案**：
- 在 `python/pyproject.toml` 中添加 `[tool.maturin]` 的 `python-packages = ["mcp_server", "cognition", "evolution", "daemon", "memory_core"]`，使 `maturin develop` 自动安装所有子包
- 或者修改 `start-mcp.ps1` 启动脚本，在启动前设置 `$env:PYTHONPATH = "$ProjectRoot\python"`
- 验收：`.mcp.json` 配置不变，FourDMem MCP Server 正常启动

---

### 方案 2：启用 CI Python 测试流水线

**问题**：`ci.yml` 第 43-58 行 Python 集成测试完全被注释。

**方案**：
- 取消注释 Linux job 中的 Python 测试步骤
- 添加 `maturin develop --release` 构建步骤
- 将 Windows job 也增加 Python 测试（可选，视 pyo3 编译速度）
- 增加 `needs: [linux]` 条件，确保 Rust 测试通过后才跑 Python
- 验收：每个 PR 自动运行 `pytest tests/ -v`，绿灯合并

---

### 方案 3：添加 Nightly 定时构建 + 依赖审计

**问题**：依赖更新（Cargo.lock / pip）可能悄悄引入安全漏洞或破坏性变更，无自动检测。

**方案**：
- 新增 `.github/workflows/nightly.yml`：
  - `schedule: cron '0 3 * * 1'`（每周一凌晨 3 点）
  - 步骤：`cargo audit` + `pip-audit` + `cargo update` 后全量测试
  - 失败自动创建 GitHub Issue 并 @维护者
- 在 `pyproject.toml` 的 `[project.optional-dependencies]` dev 中增加 `pip-audit`
- 验收：每周自动运行，漏洞检测报告推送至 Issue

---

### 方案 4：性能基准测试 CI 集成

**问题**：`criterion` 基准测试 (`crates/storage-core/benches/`, `crates/graph-core/benches/`) 只在本地运行，无回归检测。

**方案**：
- 新增 `.github/workflows/benchmark.yml`：
  - 仅在 `push: branches: [main]` 时运行
  - 使用 `benchstat` / `github-action-benchmark` 对比基准
  - 结果存入 GitHub Pages 或 artifacts
  - 性能退化 > 10% 自动标记 PR 为 `performance-regression`
- 在 `Makefile` 中添加 `bench` target：`cargo bench --workspace --exclude py-bindings`
- 验收：性能数据可追溯，退化自动报警

---

### 方案 5：自动化 Release 流水线

**问题**：版本号 (`Cargo.toml` 0.1.0, `pyproject.toml` 0.1.0) 手动维护，无自动发版。

**方案**：
- 新增 `.github/workflows/release.yml`：
  - 触发条件：`push: tags: ['v*']`
  - 步骤：`cargo publish` → `maturin build --release` → 创建 GitHub Release → 上传 wheel
  - 自动从 tag 提取版本号同步到 Cargo.toml 和 pyproject.toml（使用 `cargo-edit` 或脚本）
- 在 `Makefile` 中添加 `release` target（本地触发）
- 验收：`git tag v0.2.0 && git push --tags` 一键发版

---

### 方案 6：增强 Makefile 自动化能力

**问题**：当前 Makefile 偏薄，缺少增量构建、开发热重载、全链路检查等。

**方案**：
- 新增 target：
  - `make dev`：并行启动 `cargo watch -x check` + `maturin develop --release` + `pytest --watch` 热重载开发模式
  - `make check-all`：一键运行 `fmt + lint + test-rust + test-python + bench`
  - `make db-reset`：清理并重建 `data/vault/evidence.db` + `graph.json`
  - `make mcp-test`：启动 MCP Server 并发送 `search_memory` 测试请求
  - `make evolution-smoke`：运行认知进化烟雾测试（髓鞘化 + 范式转移 + 怪圈自指）
- 在 Makefile 开头添加 `SHELL := /bin/bash`（Linux）和现有的 Windows 兼容逻辑
- 验收：`make check-all` 一条命令覆盖全部质量门禁

---

### 方案 7：认知进化引擎的自动化调度（Cron Daemon）

**问题**：`dream_pruning` (梦境修剪)、`myelination` (髓鞘化编译)、`paradigm_shift` (范式转移监控) 有完整实现但**无自动触发机制**。

**方案**：
- 新增 `python/daemon/evolution_scheduler.py`：
  - 基于 `APScheduler` 实现：
    - 每 100 ticks 触发 `dream_pruning`（艾宾浩斯衰减）
    - 每日 3:00 AM 触发 `L1→L2 场景聚合` 和 `L2→L3 晋升检查`
    - 实时监控 `paradigm_shift.ParadigmShiftEngine.check_crisis()`
  - 将调度器注册为 MCP Server 的后台线程（在 `server.py` 的 `main()` 中启动）
- 在 `start-mcp.ps1` 中增加 `--enable-evolution-scheduler` 参数
- 验收：MCP Server 启动后，认知进化后台自动运行

---

### 方案 8：跨层引用完整性 CI 检查

**问题**：TASKS.md 中 T-2.5 要求"后台 Daemon 定期扫描跨层引用"，目前无实现。

**方案**：
- 新增 `python/daemon/integrity_checker.py`：
  - 扫描 `graph.json` 中所有 `source_l0_refs` 是否指向有效 L0 记录
  - 扫描 L2 frontmatter 中的 `l1_refs` 是否存在
  - 发现孤儿节点自动标记 `[ORPHAN]` 并归档到 `data/vault/.archive/`
- 在 CI 中添加 `make integrity-check` 步骤（`--once` 模式）
- 验收：CI 中孤儿节点检测报告 + 本地归档自动执行

---

### 方案 9：MCP Server 健康检查与自动重启

**问题**：MCP Server 进程崩溃后无自动恢复机制，客户端只能看到连接断开。

**方案**：
- 新增 `python/daemon/health_monitor.py`：
  - 定期向 MCP Server 发送 `wake_up` 探针（每 60s）
  - 连续 3 次失败则自动重启进程
  - 记录重启事件到 `data/vault/evidence.db` 供事后分析
- 在 `start-mcp.ps1` 中嵌入健康监控循环
- 或使用 `supervisord` / `NSSM`（Windows）做进程守护
- 验收：手动 kill MCP 进程后 10s 内自动恢复

---

### 方案 10：自动文档生成与 API 文档 CI

**问题**：架构文档 (`docs/`) 和 API 文档手动维护，代码变更后文档易过时。

**方案**：
- Rust 侧：在 CI 中运行 `cargo doc --workspace --no-deps`，部署到 GitHub Pages
- Python 侧：使用 `mkdocs` + `mkdocstrings[python]` 自动从 docstring 生成 API 文档
- 新增 `.github/workflows/docs.yml`：
  - `push: branches: [main]` 触发
  - 构建 `mkdocs build` + `cargo doc` → 部署到 `gh-pages` 分支
- 在 `Makefile` 中添加 `docs` target
- 验收：`https://your-org.github.io/FourDMem/` 自动更新

---

## 三、优先级排序

| 优先级 | 方案 | 理由 | 预估工时 |
|:---:|:---|:---|:---:|
| 🔴 P0 | **方案 1**：修复 MCP Server 启动 | 当前完全不可用 | 0.5h |
| 🔴 P0 | **方案 2**：启用 CI Python 测试 | Python 代码零测试保障 | 1h |
| 🟡 P1 | **方案 6**：增强 Makefile | 提升日常开发效率 | 2h |
| 🟡 P1 | **方案 7**：进化引擎自动调度 | 核心功能未自动运行 | 3h |
| 🟡 P1 | **方案 8**：跨层引用完整性检查 | 数据一致性保障 | 2h |
| 🟢 P2 | **方案 3**：Nightly + 依赖审计 | 安全与回归检测 | 1h |
| 🟢 P2 | **方案 4**：性能基准 CI | 性能回归检测 | 2h |
| 🟢 P2 | **方案 9**：MCP 健康监控 | 生产可用性保障 | 2h |
| ⚪ P3 | **方案 5**：Release 自动化 | 首版前不急需 | 2h |
| ⚪ P3 | **方案 10**：自动文档生成 | 体验提升 | 1.5h |

**总预估工时**：~17 小时，P0 + P1 约 8.5 小时

---

## 四、备注

- **FourDMem MCP 连接失败原因**：`.mcp.json` 中 `cwd` 指向 `python/`，但 `mcp_server.server` 模块未作为 Python 包安装到 venv 中（`maturin develop` 只安装了 `fourdmem` PyO3 绑定，未安装纯 Python 子包）。方案 1 通过修改 `pyproject.toml` 配置可彻底解决。
- 以上方案均**只输出方案不执行代码改动**，需确认后逐项实施。
