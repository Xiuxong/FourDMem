# Contributing to FourDMem

Thank you for your interest in contributing to FourDMem!

## Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-org>/FourDMem.git
cd FourDMem

# Initialize development environment
make init

# Run all tests
make test

# Run quality checks
make check-all
```

## Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `make test`
5. Run linting: `make lint`
6. Commit with clear messages
7. Push and create a Pull Request

## Code Style

### Rust
- Format: `cargo fmt`
- Lint: `cargo clippy -- -D warnings`
- Minimum Rust version: 1.75

### Python
- Format/Lint: `ruff check python/`
- Type check: `mypy python/`
- Minimum Python version: 3.11

## Testing

```bash
# Rust tests
make test-rust

# Python tests
make test-python

# Cognitive evolution tests
make test-evolution

# Benchmarks
make bench
```

## Architecture

See [FDRFlow.md](FDRFlow.md) for the full retrieval flow and [RIF-SQCE.md](RIF-SQCE.md) for configuration details.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
