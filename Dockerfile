# FourDMem — Cognitive Agent Memory MCP Server
# Multi-stage build: Rust compilation → Python runtime

# ── Stage 1: Build Rust bindings ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir maturin>=1.0

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY python/ python/

WORKDIR /build/python
RUN maturin develop --release 2>&1 | tail -5

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy built Python packages
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /build/python/ /app/python/

# Copy data and config directories (will be overridden by volume mount)
COPY data/ /app/data/ 2>/dev/null || true

# Install runtime Python dependencies
COPY python/pyproject.toml /app/python/
RUN pip install --no-cache-dir mcp pydantic watchdog jinja2 loguru networkx scikit-learn gitpython litellm tiktoken

# Data directory for persistent storage
VOLUME ["/app/data"]

# MCP server runs on stdio
ENV PYTHONPATH=/app/python
ENTRYPOINT ["python", "-m", "mcp_server.server"]
