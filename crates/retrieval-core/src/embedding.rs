//! Lightweight text embedding via character trigram hashing
//!
//! This module provides a fast, zero-dependency text embedding method
//! suitable for semantic similarity search. It uses **character trigram
//! hashing with random projection** — similar texts produce similar
//! vectors without requiring any ML model or external dependency.
//!
//! # Algorithm
//!
//! 1. Slide a 3-character window over the input text (trigrams).
//! 2. Hash each trigram to a `u64` using `std::hash::Hasher`.
//! 3. Use the hash as a seed to generate `dim` pseudo-random components
//!    via `sin(hash * i * PHASE)` — a deterministic random projection.
//! 4. Accumulate all trigram contributions into a single vector.
//! 5. L2-normalize the result.
//!
//! The resulting vectors have unit length, so cosine similarity equals
//! the dot product.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Default embedding dimension. 768 matches bge-base-en-v1.5 output.
pub const DEFAULT_EMBEDDING_DIM: usize = 768;

/// Phase constant for the pseudo-random projection.
/// An irrational number ensures good distribution across dimensions.
const PHASE: f64 = 1.618_033_988_749_895; // golden ratio

/// **DEPRECATED**: This trigram-hash embedding produces low-quality vectors
/// that are incompatible with real semantic embeddings (e.g. bge-small-zh-v1.5).
/// Use `ingest_with_embedding` / `add_fact_with_embedding` / `query_with_embedding`
/// with a Python-computed embedding instead.
///
/// Kept only for backward compatibility with existing tests. Will be removed in v0.2.
///
/// Returns a zero vector for empty input.
#[deprecated(since = "0.2", note = "use Python bge-base-en-v1.5 embeddings instead")]
pub fn embed_text(text: &str, dim: usize) -> Vec<f32> {
    let mut vec = vec![0.0f32; dim];

    if text.is_empty() || dim == 0 {
        return vec;
    }

    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 3 {
        // For very short text, hash the whole thing as one "trigram"
        let hash = hash_str(text);
        accumulate(hash, dim, &mut vec);
    } else {
        for window in chars.windows(3) {
            let trigram: String = window.iter().collect();
            let hash = hash_str(&trigram);
            accumulate(hash, dim, &mut vec);
        }
    }

    // L2 normalize
    l2_normalize(&mut vec);
    vec
}

/// Compute cosine similarity between two embedding vectors.
///
/// Both vectors should be L2-normalized (as produced by `embed_text`),
/// in which case this is just the dot product. The function handles
/// unnormalized vectors correctly as well.
#[deprecated(since = "0.2", note = "use Python bge-base-en-v1.5 embeddings instead")]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Hash a string to a u64 using DefaultHasher.
fn hash_str(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Accumulate a trigram's contribution into the vector using
/// pseudo-random projection seeded by the trigram hash.
fn accumulate(hash: u64, dim: usize, vec: &mut [f32]) {
    let h = hash as f64;
    for i in 0..dim {
        // sin-based pseudo-random projection: deterministic but well-distributed
        let val = (h * (i as f64 + 1.0) * PHASE).sin();
        vec[i] += val as f32;
    }
}

/// L2-normalize a vector in-place. No-op if the vector is zero.
fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

/// Find the top-k most semantically similar fact pairs across different tag domains.
///
/// Given a list of nodes with their embeddings and tag sets, this function
/// identifies pairs from **different** domains (defined by primary tag) that
/// have high cosine similarity — suggesting a potential cross-domain analogy.
///
/// Returns `Vec<(idx_a, idx_b, similarity)>` sorted by similarity descending,
/// limited to `top_k` results.
pub fn cross_domain_analogies(
    nodes: &[(String, Vec<f32>)],   // (label, embedding)
    tags: &[Vec<String>],           // primary tags per node (parallel to nodes)
    top_k: usize,
) -> Vec<(usize, usize, f32)> {
    if nodes.len() != tags.len() || nodes.is_empty() {
        return Vec::new();
    }

    // Determine the primary domain for each node (first tag, or "untagged")
    let domains: Vec<&str> = tags
        .iter()
        .map(|t| t.first().map(|s| s.as_str()).unwrap_or("untagged"))
        .collect();

    let mut pairs: Vec<(usize, usize, f32)> = Vec::new();

    // Compare all cross-domain pairs
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            // Only consider pairs from different domains
            if domains[i] == domains[j] {
                continue;
            }
            let sim = cosine_sim(&nodes[i].1, &nodes[j].1);
            if sim > 0.0 {
                pairs.push((i, j, sim));
            }
        }
    }

    // Sort by similarity descending
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(top_k);
    pairs
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_produces_unit_vector() {
        let v = embed_text("hello world", DEFAULT_EMBEDDING_DIM);
        assert_eq!(v.len(), DEFAULT_EMBEDDING_DIM);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm should be ~1.0, got {}", norm);
    }

    #[test]
    fn test_similar_texts_high_similarity() {
        let v1 = embed_text("Rust borrow checker prevents data races", DEFAULT_EMBEDDING_DIM);
        let v2 = embed_text("Rust borrow checker ensures memory safety", DEFAULT_EMBEDDING_DIM);
        let v3 = embed_text("Python garbage collection handles memory", DEFAULT_EMBEDDING_DIM);

        let sim_related = cosine_sim(&v1, &v2);
        let sim_unrelated = cosine_sim(&v1, &v3);

        assert!(
            sim_related > sim_unrelated,
            "related texts ({}) should be more similar than unrelated ({})",
            sim_related,
            sim_unrelated
        );
    }

    #[test]
    fn test_identical_texts_similarity_one() {
        let v = embed_text("identical", DEFAULT_EMBEDDING_DIM);
        let sim = cosine_sim(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "self-similarity should be 1.0, got {}", sim);
    }

    #[test]
    fn test_empty_text_returns_zero_vector() {
        let v = embed_text("", DEFAULT_EMBEDDING_DIM);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_short_text_works() {
        let v = embed_text("ab", DEFAULT_EMBEDDING_DIM);
        assert_eq!(v.len(), DEFAULT_EMBEDDING_DIM);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_different_dim() {
        let v = embed_text("test", 64);
        assert_eq!(v.len(), 64);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_sim_zero_vectors() {
        let a = vec![0.0f32; 128];
        let b = vec![0.0f32; 128];
        assert_eq!(cosine_sim(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        let mut a = vec![0.0f32; 4];
        let mut b = vec![0.0f32; 4];
        a[0] = 1.0;
        b[1] = 1.0;
        assert!(cosine_sim(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_deterministic() {
        let v1 = embed_text("deterministic test", 128);
        let v2 = embed_text("deterministic test", 128);
        assert_eq!(v1, v2, "embedding should be deterministic");
    }

    #[test]
    fn test_cross_domain_analogies() {
        let nodes = vec![
            ("rust_safety".to_string(), embed_text("borrow checker memory safety", 64)),
            ("rust_perf".to_string(), embed_text("zero cost abstractions performance", 64)),
            ("python_safety".to_string(), embed_text("garbage collector memory safety", 64)),
            ("python_perf".to_string(), embed_text("interpreter overhead performance", 64)),
        ];
        let tags = vec![
            vec!["rust".to_string(), "safety".to_string()],
            vec!["rust".to_string(), "performance".to_string()],
            vec!["python".to_string(), "safety".to_string()],
            vec!["python".to_string(), "performance".to_string()],
        ];

        let analogies = cross_domain_analogies(&nodes, &tags, 10);
        // Should find cross-domain pairs (rust↔python), not same-domain pairs
        for &(i, j, _sim) in &analogies {
            assert_ne!(
                tags[i][0], tags[j][0],
                "should only return cross-domain pairs"
            );
        }
        // Should find at least the safety↔safety and perf↔perf cross-domain pairs
        assert!(!analogies.is_empty(), "should find cross-domain analogies");
    }

    #[test]
    fn test_cross_domain_empty() {
        let analogies = cross_domain_analogies(&[], &[], 5);
        assert!(analogies.is_empty());
    }
}
