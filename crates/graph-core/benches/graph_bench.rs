use criterion::{black_box, criterion_group, criterion_main, Criterion};
use graph_core::{L1Graph, NodeAttr, EdgeAttr, VersionTree};

fn bench_graph_add_nodes(c: &mut Criterion) {
    c.bench_function("graph_add_1000_nodes", |b| {
        b.iter(|| {
            let mut g = L1Graph::new();
            for i in 0..1000 {
                g.add_node(NodeAttr::new(black_box(&format!("node {}", i))));
            }
            g
        })
    });
}

fn bench_graph_conflict_traversal(c: &mut Criterion) {
    let mut g = L1Graph::new();
    let nodes: Vec<_> = (0..100).map(|i| g.add_node(NodeAttr::new(format!("n{}", i)))).collect();
    // Add some conflict edges
    for i in 0..50 {
        let _ = g.add_edge(nodes[i], nodes[i + 50], EdgeAttr::contradicts(0.8));
    }

    c.bench_function("graph_find_conflicts_100_nodes", |b| {
        b.iter(|| {
            for ni in &nodes {
                black_box(g.find_conflicts(*ni));
            }
        })
    });
}

fn bench_version_tree_query(c: &mut Criterion) {
    let mut tree = VersionTree::new();
    // Insert 1000 versions for one entity
    for i in 0..1000 {
        tree.insert("entity", format!("hash_{}", i), serde_json::json!(i));
    }

    c.bench_function("version_tree_query_1000", |b| {
        b.iter(|| {
            black_box(tree.history(black_box("entity")));
        })
    });
}

criterion_group!(benches, bench_graph_add_nodes, bench_graph_conflict_traversal, bench_version_tree_query);
criterion_main!(benches);
