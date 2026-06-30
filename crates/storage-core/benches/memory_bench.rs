use criterion::{black_box, criterion_group, criterion_main, Criterion};
use storage_core::MemoryCore;

fn bench_memory_query(c: &mut Criterion) {
    let mut mc = MemoryCore::new_in_memory().unwrap();
    // Pre-populate with facts
    for i in 0..100 {
        mc.add_fact(&format!("fact number {} about testing", i), None);
    }

    c.bench_function("memory_query_100_facts", |b| {
        b.iter(|| {
            mc.query(&storage_core::QueryRequest::simple(black_box("testing"))).unwrap()
        })
    });
}

fn bench_memory_ingest(c: &mut Criterion) {
    let mut mc = MemoryCore::new_in_memory().unwrap();
    let mut counter = 0u64;

    c.bench_function("memory_ingest_evidence", |b| {
        b.iter(|| {
            counter += 1;
            mc.ingest_evidence(
                black_box("bench-ws"),
                black_box("bench-model"),
                black_box("bench-session"),
                black_box("user"),
                black_box(&format!("evidence {}", counter)),
                black_box(&serde_json::Value::Null),
            ).unwrap();
        })
    });
}

criterion_group!(benches, bench_memory_query, bench_memory_ingest);
criterion_main!(benches);
