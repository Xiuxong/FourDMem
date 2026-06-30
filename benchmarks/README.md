# FourDMem Benchmarks

Reproducible benchmark suite for FourDMem memory system.

## Running benchmarks

```bash
# From project root
cd python
python -m benchmarks.run_all

# Individual benchmarks
python -m benchmarks.locomo_bench
python -m benchmarks.longmemeval_bench
python -m benchmarks.synthetic_recall
```

## Benchmarks

| Benchmark | What it measures | Dataset |
|-----------|-----------------|---------|
| `synthetic_recall` | Basic recall accuracy on injected facts | Built-in synthetic |
| `locomo_bench` | Long conversation memory recall | LoCoMo (auto-download) |
| `longmemeval_bench` | Long-term memory evaluation | LongMemEval (auto-download) |

## Metrics

- **R@K**: Recall at K — fraction of questions where the correct answer is in top-K results
- **Precision@K**: Fraction of top-K results that are relevant
- **MRR**: Mean Reciprocal Rank of the correct answer
- **Latency P50/P99**: Query latency at 50th/99th percentile

## Reproducing

All benchmarks use deterministic seeds. Run with:
```bash
python -m benchmarks.run_all --seed 42 --output results/
```
