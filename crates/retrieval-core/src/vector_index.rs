//! Vector index for embedding-based similarity search.
//!
//! Two backends:
//! - **`vector-index` feature enabled**: Uses `usearch` HNSW for high-performance
//!   approximate nearest-neighbour search (requires C++17 toolchain).
//! - **`vector-index` feature disabled** (default): Uses a pure-Rust flat
//!   brute-force index. Functional on all platforms including MSVC.
//!
//! Both backends expose the same `VectorIndex` API. The `search` method
//! returns `(key, similarity_score)` pairs sorted by similarity descending
//! (higher = more similar).

// ── usearch HNSW backend ──────────────────────────────────────────────────────

#[cfg(feature = "vector-index")]
mod inner {
    use usearch::{Index, IndexOptions, MetricKind};

    /// HNSW vector index backed by `usearch`.
    pub struct VectorIndex {
        index: Index,
        dim: usize,
    }

    impl VectorIndex {
        pub fn new(dim: usize) -> Result<Self, String> {
            let mut opts = IndexOptions::default();
            opts.dimensions = dim;
            opts.metric = MetricKind::Cos;
            let index = Index::new(&opts).map_err(|e| e.to_string())?;
            Ok(Self { index, dim })
        }

        pub fn add(&mut self, key: u64, vector: &[f32]) -> Result<(), String> {
            assert_eq!(vector.len(), self.dim);
            self.index
                .add(key, vector)
                .map_err(|e| e.to_string())
        }

        /// Search for the `k` nearest neighbours.
        ///
        /// Returns `(key, similarity)` pairs sorted by similarity descending.
        /// usearch returns cosine distance; we convert: similarity = 1 / (1 + distance).
        pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, String> {
            assert_eq!(query.len(), self.dim);
            let results = self.index.search(query, k).map_err(|e| e.to_string())?;
            Ok(results
                .keys
                .into_iter()
                .zip(results.distances.into_iter())
                .map(|(key, dist)| (key, 1.0 / (1.0 + dist)))
                .collect())
        }

        pub fn remove(&mut self, key: u64) -> Result<(), String> {
            self.index.remove(key).map_err(|e| e.to_string())
        }

        pub fn len(&self) -> usize {
            self.index.size()
        }

        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }

        /// Save index to disk.
        pub fn save(&self, path: &str) -> Result<(), String> {
            self.index.save(path).map_err(|e| e.to_string())
        }

        /// Load index from disk.
        pub fn load(path: &str, dim: usize) -> Result<Self, String> {
            let mut opts = IndexOptions::default();
            opts.dimensions = dim;
            opts.metric = MetricKind::Cos;
            let index = Index::new(&opts).map_err(|e| e.to_string())?;
            index.load(path).map_err(|e| e.to_string())?;
            Ok(Self { index, dim })
        }
    }
}

// ── Flat brute-force backend (pure Rust, MSVC-compatible) ─────────────────────

#[cfg(not(feature = "vector-index"))]
mod inner {
    /// Flat brute-force vector index.
    ///
    /// Stores all vectors in a `Vec` and scans linearly on search.
    /// Suitable for up to ~100K vectors; beyond that, use the `vector-index`
    /// feature for HNSW acceleration.
    pub struct VectorIndex {
        dim: usize,
        vectors: Vec<(u64, Vec<f32>)>,
    }

    impl VectorIndex {
        pub fn new(dim: usize) -> Result<Self, String> {
            Ok(Self {
                dim,
                vectors: Vec::new(),
            })
        }

        pub fn add(&mut self, key: u64, vector: &[f32]) -> Result<(), String> {
            if vector.len() != self.dim {
                return Err(format!(
                    "dimension mismatch: expected {}, got {}",
                    self.dim,
                    vector.len()
                ));
            }
            self.vectors.push((key, vector.to_vec()));
            Ok(())
        }

        /// Search for the `k` most similar vectors.
        ///
        /// Returns `(key, cosine_similarity)` pairs sorted by similarity descending.
        pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, String> {
            if query.len() != self.dim {
                return Err(format!(
                    "dimension mismatch: expected {}, got {}",
                    self.dim,
                    query.len()
                ));
            }

            let mut scored: Vec<(u64, f32)> = self
                .vectors
                .iter()
                .map(|(key, vec)| (*key, cosine_similarity(query, vec)))
                .collect();

            // Sort by similarity descending (highest first)
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(k);
            Ok(scored)
        }

        pub fn remove(&mut self, key: u64) -> Result<(), String> {
            let before = self.vectors.len();
            self.vectors.retain(|(k, _)| *k != key);
            if self.vectors.len() == before {
                Err(format!("key {} not found", key))
            } else {
                Ok(())
            }
        }

        pub fn len(&self) -> usize {
            self.vectors.len()
        }

        pub fn is_empty(&self) -> bool {
            self.vectors.is_empty()
        }

        /// Save flat index to a JSON file.
        pub fn save(&self, path: &str) -> Result<(), String> {
            use std::fs;
            use std::io::Write;
            let mut buf = Vec::new();
            write!(buf, "{{\"dim\":{},\"vectors\":[", self.dim)
                .map_err(|e| e.to_string())?;
            for (i, (key, vec)) in self.vectors.iter().enumerate() {
                if i > 0 { write!(buf, ",").ok(); }
                write!(buf, "{{\"k\":{},\"v\":[", key).map_err(|e| e.to_string())?;
                for (j, v) in vec.iter().enumerate() {
                    if j > 0 { write!(buf, ",").ok(); }
                    write!(buf, "{}", v).ok();
                }
                write!(buf, "]}}").ok();
            }
            write!(buf, "]}}").map_err(|e| e.to_string())?;
            fs::write(path, buf).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Load flat index from a JSON file.
        pub fn load(path: &str, dim: usize) -> Result<Self, String> {
            use std::fs;
            let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
            let json: serde_json::Value = serde_json::from_str(&content)
                .map_err(|e| e.to_string())?;
            let file_dim = json["dim"].as_u64().unwrap_or(dim as u64) as usize;
            let mut vectors = Vec::new();
            if let Some(arr) = json["vectors"].as_array() {
                for item in arr {
                    let key = item["k"].as_u64().unwrap_or(0);
                    let vec: Vec<f32> = item["v"].as_array()
                        .map(|a| a.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
                        .unwrap_or_default();
                    vectors.push((key, vec));
                }
            }
            Ok(Self { dim: file_dim, vectors })
        }
    }

    /// Compute cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }
}

// ── Re-export ─────────────────────────────────────────────────────────────────

pub use inner::VectorIndex;

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_index_new() {
        let idx = VectorIndex::new(4).unwrap();
        assert_eq!(idx.len(), 0);
        assert!(idx.is_empty());
    }

    #[test]
    fn test_flat_index_add_and_len() {
        let mut idx = VectorIndex::new(4).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 2);
        assert!(!idx.is_empty());
    }

    #[test]
    fn test_flat_index_search_ranking() {
        let mut idx = VectorIndex::new(4).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        idx.add(3, &[0.0, 0.0, 1.0, 0.0]).unwrap();

        // Query closest to key 1
        let results = idx.search(&[1.0, 0.1, 0.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 1, "key 1 should be top result");
        // All similarities should be descending
        for i in 1..results.len() {
            assert!(
                results[i].1 <= results[i - 1].1,
                "results should be sorted by similarity descending"
            );
        }
    }

    #[test]
    fn test_flat_index_search_top_k() {
        let mut idx = VectorIndex::new(4).unwrap();
        for i in 0..10 {
            let mut v = vec![0.0f32; 4];
            v[i % 4] = 1.0;
            idx.add(i as u64, &v).unwrap();
        }
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_flat_index_remove() {
        let mut idx = VectorIndex::new(4).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 2);

        idx.remove(1).unwrap();
        assert_eq!(idx.len(), 1);

        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_flat_index_remove_nonexistent() {
        let mut idx = VectorIndex::new(4).unwrap();
        assert!(idx.remove(999).is_err());
    }

    #[test]
    fn test_flat_index_dimension_mismatch() {
        let mut idx = VectorIndex::new(4).unwrap();
        assert!(idx.add(1, &[1.0, 0.0]).is_err()); // dim 2 != 4
        assert!(idx.search(&[1.0, 0.0], 1).is_err());
    }

    #[test]
    fn test_flat_index_similarity_range() {
        let mut idx = VectorIndex::new(4).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        let sim = results[0].1;
        assert!(sim >= 0.0 && sim <= 1.0, "similarity should be in [0,1], got {}", sim);
        assert!((sim - 1.0).abs() < 1e-5, "self-similarity should be ~1.0");
    }

    #[test]
    fn test_flat_index_empty_search() {
        let idx = VectorIndex::new(4).unwrap();
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }
}
