use criterion::{black_box, criterion_group, criterion_main, Criterion};
use storage_core::L0Store;

fn bench_l0_append(c: &mut Criterion) {
    let store = L0Store::open_memory().unwrap();
    let mut counter = 0u64;

    c.bench_function("l0_append_1000", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                counter += 1;
                store.append(
                    black_box("bench-ws"),
                    black_box("bench-model"),
                    black_box("bench-session"),
                    black_box("user"),
                    black_box(&format!("evidence item {}", counter)),
                    black_box(&serde_json::json!({"i": counter})),
                ).unwrap();
            }
        })
    });
}

fn bench_l0_search(c: &mut Criterion) {
    let store = L0Store::open_memory().unwrap();
    // Pre-populate with 1000 items
    for i in 0..1000 {
        store.append("bench-ws", "bench-model", "s", "user", &format!("search benchmark item {}", i), &serde_json::json!(null)).unwrap();
    }

    c.bench_function("l0_search_1000_items", |b| {
        b.iter(|| {
            store.search(black_box("benchmark"), black_box(10), None).unwrap()
        })
    });
}

criterion_group!(benches, bench_l0_append, bench_l0_search);
criterion_main!(benches);
