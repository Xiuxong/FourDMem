.PHONY: init test test-rust test-python test-evolution run-mcp fmt lint clean help dev check-all db-reset mcp-test evolution-smoke integrity-check bench

# Detect OS for cross-platform venv activation
ifeq ($(OS),Windows_NT)
    VENV_ACTIVATE = .venv/Scripts/activate
    PYTHON = .venv/Scripts/python.exe
    PIP = .venv/Scripts/pip.exe
    RMDIR = if exist $(1) rmdir /s /q $(1)
else
    VENV_ACTIVATE = . .venv/bin/activate &&
    PYTHON = .venv/bin/python
    PIP = .venv/bin/pip
    RMDIR = rm -rf $(1)
endif

# ── Default target ─────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────
init: ## Initialize Rust + Python mixed environment
	@echo "==> Installing Rust dependencies..."
	cargo check
	@echo "==> Setting up Python virtual environment..."
	cd python && python -m venv .venv
	cd python && $(VENV_ACTIVATE) pip install --upgrade pip && pip install maturin
	@echo "==> Building PyO3 bindings..."
	cd python && $(VENV_ACTIVATE) maturin develop --release
	@echo "==> Installing Python dev dependencies..."
	cd python && $(VENV_ACTIVATE) pip install -e ".[dev]"
	@echo "==> Done! Run 'make test' to verify."

# ── Testing ────────────────────────────────────────────
test: test-rust test-python ## Run all tests with coverage

test-rust: ## Run Rust unit tests
	@echo "==> Running Rust tests..."
	cargo test --workspace

test-python: ## Run Python integration tests
	@echo "==> Running Python tests..."
	cd python && $(VENV_ACTIVATE) pytest tests/ -v

test-evolution: ## Run cognitive evolution tests
	@echo "==> Running evolution tests..."
	cd python && $(VENV_ACTIVATE) pytest tests/test_evolution.py -v

# ── Run ────────────────────────────────────────────────
run-mcp: ## Start MCP Server (with cold-start wake-up)
	@echo "==> Starting FourDMem MCP Server..."
	cd python && $(VENV_ACTIVATE) python -m mcp_server.server

# ── Code Quality ───────────────────────────────────────
fmt: ## Format all code (Rust + Python)
	@echo "==> Formatting Rust code..."
	cargo fmt --all
	@echo "==> Formatting Python code..."
	cd python && $(VENV_ACTIVATE) ruff format .

lint: ## Lint all code (Clippy + Ruff)
	@echo "==> Running Clippy..."
	cargo clippy --workspace -- -D warnings
	@echo "==> Running Ruff..."
	cd python && $(VENV_ACTIVATE) ruff check .

# ── Cleanup ────────────────────────────────────────────
clean: ## Clean build artifacts
	@echo "==> Cleaning Rust build artifacts..."
	cargo clean
	@echo "==> Cleaning Python cache..."
	-powershell -Command "Get-ChildItem -Path python -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	-powershell -Command "Get-ChildItem -Path python -Recurse -Directory -Filter .pytest_cache | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	-rmdir /s /q python\.venv 2>nul || rm -rf python/.venv 2>/dev/null || true
	@echo "==> Done!"

# ── Development ─────────────────────────────────────
dev: ## Hot-reload dev mode (Rust check + maturin + pytest watch)
	@echo "==> Starting hot-reload development mode..."
	@echo "    Rust: cargo watch -x check"
	@echo "    Python: maturin develop + pytest --watch"
	cargo watch -x check &
	cd python && $(VENV_ACTIVATE) maturin develop --release
	cd python && $(VENV_ACTIVATE) pytest tests/ -v --tb=short

check-all: fmt lint test bench ## Run all quality gates (fmt + lint + test + bench)
	@echo "==> All checks passed!"

# ── Database ────────────────────────────────────────
db-reset: ## Reset L0 database and graph
	@echo "==> Resetting database..."
	-del /q data\vault\evidence.db 2>nul || rm -f data/vault/evidence.db 2>/dev/null || true
	-del /q data\vault\evidence.db-shm 2>nul || rm -f data/vault/evidence.db-shm 2>/dev/null || true
	-del /q data\vault\evidence.db-wal 2>nul || rm -f data/vault/evidence.db-wal 2>/dev/null || true
	-del /q data\graph.json 2>nul || rm -f data/graph.json 2>/dev/null || true
	@echo "==> Database reset. Run 'make run-mcp' to recreate."

# ── MCP Testing ─────────────────────────────────────
mcp-test: ## Test MCP Server startup and basic tools
	@echo "==> Testing MCP Server..."
	cd python && $(PYTHON) -c "from mcp_server.server import mcp; tools = [t.name for t in mcp._tool_manager._tools.values()]; print(f'Registered {len(tools)} tools:', tools); assert len(tools) >= 8, f'Expected >=8 tools, got {len(tools)}'; print('MCP Server OK')"

# ── Evolution Smoke Test ────────────────────────────
evolution-smoke: ## Smoke test cognitive evolution engine
	@echo "==> Evolution smoke test..."
	cd python && $(PYTHON) -c "from evolution.strange_loop import ObserverNode; from evolution.paradigm_shift import ParadigmShiftEngine; from evolution.myelination import MyelinationTracker; from cognition.dream import DreamPruner; from cognition.salience import SalienceDetector; from cognition.dedup import SemanticDeduplicator; print('All evolution modules imported OK'); o = ObserverNode(); p = ParadigmShiftEngine(); m = MyelinationTracker(); d = DreamPruner(); s = SalienceDetector(); dd = SemanticDeduplicator(); print(f'Observer: {o.get_observation_summary()}'); print(f'Myelination: {m.get_stats()}'); print('Evolution smoke test PASSED')"

# ── Integrity Check ─────────────────────────────────
integrity-check: ## Check cross-layer reference integrity
	@echo "==> Running integrity check..."
	cd python && $(PYTHON) -m daemon.integrity_checker --once

# ── Benchmarks ───────────────────────────────────────
bench: ## Run Rust benchmarks
	@echo "==> Running Rust benchmarks..."
	cargo bench --workspace --exclude py-bindings 2>/dev/null || echo "Benchmarks skipped (no bench targets found)"
