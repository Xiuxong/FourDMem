//! Reciprocal Rank Fusion (RRF) reranker
//!
//! Fuses multiple ranked result lists (from vector, full-text, and graph
//! retrieval) into a single ranked list using the RRF formula:
//!
//! ```text
//! RRF_score(doc) = Σ  1 / (k + rank_i(doc))
//! ```
//!
//! where `k` is a smoothing constant (default 40) and `rank_i` is the
//! 1-based rank of the document in list `i`.

use std::collections::HashMap;

// ── Types ─────────────────────────────────────────────────────────────────────

/// A single item in a ranked result list.
#[derive(Debug, Clone)]
pub struct RankedItem {
    /// Unique document / fact identifier.
    pub doc_id: String,
    /// Original retrieval score (informational only — RRF uses rank, not score).
    pub score: f64,
}

/// Reciprocal Rank Fusion fuser.
pub struct RrfFuser {
    /// Smoothing constant. Higher values give more weight to lower-ranked
    /// documents. Default: 40 (standard in the literature).
    pub k: f64,
}

impl Default for RrfFuser {
    fn default() -> Self {
        Self { k: 40.0 }
    }
}

impl RrfFuser {
    /// Create a fuser with the default k=40.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a fuser with a custom smoothing constant.
    pub fn with_k(k: f64) -> Self {
        Self { k }
    }

    /// Fuse multiple ranked lists into a single ranked list.
    ///
    /// Each input list is assumed to be sorted by relevance (index 0 = best).
    /// The output is sorted by fused RRF score descending.
    pub fn fuse(&self, lists: &[Vec<RankedItem>]) -> Vec<RankedItem> {
        let mut scores: HashMap<String, f64> = HashMap::new();

        for list in lists {
            for (rank, item) in list.iter().enumerate() {
                let rrf_score = 1.0 / (self.k + (rank as f64) + 1.0); // rank is 0-based, convert to 1-based
                *scores.entry(item.doc_id.clone()).or_insert(0.0) += rrf_score;
            }
        }

        let mut fused: Vec<RankedItem> = scores
            .into_iter()
            .map(|(doc_id, score)| RankedItem { doc_id, score })
            .collect();

        // Sort by RRF score descending
        fused.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        fused
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_list(ids: &[&str]) -> Vec<RankedItem> {
        ids.iter()
            .map(|id| RankedItem {
                doc_id: id.to_string(),
                score: 1.0,
            })
            .collect()
    }

    #[test]
    fn test_single_list_passthrough() {
        let fuser = RrfFuser::new();
        let list = make_list(&["a", "b", "c"]);
        let result = fuser.fuse(&[list]);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].doc_id, "a");
        assert_eq!(result[1].doc_id, "b");
        assert_eq!(result[2].doc_id, "c");
    }

    #[test]
    fn test_two_lists_boost_overlapping() {
        let fuser = RrfFuser::new();
        let l1 = make_list(&["a", "b", "c"]);
        let l2 = make_list(&["b", "a", "d"]);
        let result = fuser.fuse(&[l1, l2]);

        // "a" and "b" appear in both lists with symmetric ranks,
        // so they have identical RRF scores and both rank above "c" and "d"
        let top2: Vec<&str> = result[0..2].iter().map(|r| r.doc_id.as_str()).collect();
        assert!(top2.contains(&"a"));
        assert!(top2.contains(&"b"));
        // c and d each appear once and rank below the overlapping docs
        assert!(result[2].score < result[0].score);
    }

    #[test]
    fn test_empty_lists() {
        let fuser = RrfFuser::new();
        let result = fuser.fuse(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_empty_sublists() {
        let fuser = RrfFuser::new();
        let result = fuser.fuse(&[vec![], vec![]]);
        assert!(result.is_empty());
    }
}
