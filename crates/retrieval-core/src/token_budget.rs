//! Token budget layer quota allocator
//!
//! Enforces the 1500-token budget with per-layer quotas:
//!
//! ```text
//! L3 (Persona/Core)      : 20%  = 300 tokens
//! L2 (Scenario Blocks)   : 40%  = 600 tokens
//! L1 (Atomic Facts)      : 30%  = 450 tokens
//! L0 (Raw Evidence)      : 10%  = 150 tokens
//! ```
//!
//! When a layer has fewer items than its quota, unused tokens are
//! redistributed to neighbouring layers (lower layers first).

use serde::{Deserialize, Serialize};

// ── Constants ──────────────────────────────────────────────────────────────────

/// Default total token budget for a single query response.
pub const DEFAULT_BUDGET: usize = 1500;

/// Quota ratios for each layer (L0–L3). Must sum to 1.0.
pub const L3_RATIO: f64 = 0.20;
pub const L2_RATIO: f64 = 0.40;
pub const L1_RATIO: f64 = 0.30;
pub const L0_RATIO: f64 = 0.10;

// ── Types ──────────────────────────────────────────────────────────────────────

/// A candidate memory item with its abstraction layer and pre-computed score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetCandidate {
    /// Unique identifier (doc_id, node_id, etc.).
    pub id: String,
    /// Abstraction layer: 0, 1, 2, or 3.
    pub layer: u8,
    /// Pre-computed relevance score (higher = more relevant).
    pub score: f64,
    /// Estimated token count for this item's content.
    pub token_estimate: usize,
}

/// Result of a budget allocation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAllocation {
    /// Items selected for inclusion, ordered by layer then score.
    pub selected: Vec<BudgetCandidate>,
    /// Total tokens consumed by selected items.
    pub total_tokens: usize,
    /// Per-layer token usage: `(layer, tokens_used, quota)`.
    pub layer_usage: Vec<(u8, usize, usize)>,
}

// ── Allocator ──────────────────────────────────────────────────────────────────

/// Token budget allocator with per-layer quotas and overflow redistribution.
pub struct TokenBudget {
    /// Total token budget.
    pub budget: usize,
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self {
            budget: DEFAULT_BUDGET,
        }
    }
}

impl TokenBudget {
    /// Create a new allocator with the default 1500-token budget.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new allocator with a custom budget.
    pub fn with_budget(budget: usize) -> Self {
        Self { budget }
    }

    /// Compute per-layer quotas from the total budget.
    fn quotas(&self) -> [usize; 4] {
        [
            (self.budget as f64 * L0_RATIO) as usize, // L0
            (self.budget as f64 * L1_RATIO) as usize, // L1
            (self.budget as f64 * L2_RATIO) as usize, // L2
            (self.budget as f64 * L3_RATIO) as usize, // L3
        ]
    }

    /// Allocate candidates within the token budget.
    ///
    /// Candidates are grouped by layer, sorted by score descending within each
    /// layer, then greedily filled up to the layer's quota. Unused tokens from
    /// a layer are redistributed to the next lower layer (L3→L2→L1→L0).
    pub fn allocate(&self, candidates: &[BudgetCandidate]) -> BudgetAllocation {
        let mut quotas = self.quotas();

        // Group candidates by layer, sorted by score descending.
        let mut by_layer: [Vec<&BudgetCandidate>; 4] = [vec![], vec![], vec![], vec![]];
        for c in candidates {
            if c.layer <= 3 {
                by_layer[c.layer as usize].push(c);
            }
        }
        for layer in by_layer.iter_mut() {
            layer.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Redistribute quotas from empty layers downward (L3→L2→L1→L0).
        // Upper empty layers spill their quota to the next lower layer only.
        for layer_idx in (1..=3).rev() {
            if by_layer[layer_idx].is_empty() && quotas[layer_idx] > 0 {
                quotas[layer_idx - 1] += quotas[layer_idx];
                quotas[layer_idx] = 0;
            }
        }
        // Greedy fill per layer, tracking overflow for redistribution.
        let mut selected: Vec<BudgetCandidate> = Vec::new();
        let mut layer_tokens = [0usize; 4];

        // Process L3 → L2 → L1 → L0. Unused tokens flow downward.
        for layer_idx in (0..=3).rev() {
            let mut remaining = quotas[layer_idx];
            for item in by_layer[layer_idx].iter() {
                if item.token_estimate <= remaining {
                    remaining -= item.token_estimate;
                    layer_tokens[layer_idx] += item.token_estimate;
                    selected.push((*item).clone());
                }
            }
            // Redistribute unused tokens to the next lower layer.
            if layer_idx > 0 && remaining > 0 {
                quotas[layer_idx - 1] += remaining;
            }
        }

        let total_tokens: usize = layer_tokens.iter().sum();

        let layer_usage: Vec<(u8, usize, usize)> = (0..=3)
            .map(|i| (i as u8, layer_tokens[i], quotas[i]))
            .collect();

        BudgetAllocation {
            selected,
            total_tokens,
            layer_usage,
        }
    }

    /// Allocate candidates without RIF-U layer quotas (flat allocation).
    ///
    /// Candidates are sorted by score descending and returned up to
    /// the budget limit. No layer-quota enforcement — used for ablation.
    pub fn allocate_flat(&self, candidates: &[BudgetCandidate]) -> BudgetAllocation {
        let mut sorted: Vec<BudgetCandidate> = candidates.to_vec();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected: Vec<BudgetCandidate> = Vec::new();
        let mut total_tokens: usize = 0;
        for item in sorted {
            if total_tokens + item.token_estimate <= self.budget {
                total_tokens += item.token_estimate;
                selected.push(item);
            }
        }

        BudgetAllocation {
            selected,
            total_tokens,
            layer_usage: vec![],
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(id: &str, layer: u8, score: f64, tokens: usize) -> BudgetCandidate {
        BudgetCandidate {
            id: id.to_string(),
            layer,
            score,
            token_estimate: tokens,
        }
    }

    #[test]
    fn test_default_quotas() {
        let budget = TokenBudget::new();
        let q = budget.quotas();
        assert_eq!(q[0], 150); // L0: 10%
        assert_eq!(q[1], 450); // L1: 30%
        assert_eq!(q[2], 600); // L2: 40%
        assert_eq!(q[3], 300); // L3: 20%
        // Total should be close to 1500 (integer truncation may lose ≤3 tokens)
        assert!(q.iter().sum::<usize>() >= 1497);
    }

    #[test]
    fn test_empty_input() {
        let budget = TokenBudget::new();
        let alloc = budget.allocate(&[]);
        assert!(alloc.selected.is_empty());
        assert_eq!(alloc.total_tokens, 0);
    }

    #[test]
    fn test_single_layer_under_quota() {
        let budget = TokenBudget::new();
        let candidates = vec![
            make_candidate("a", 2, 0.9, 100),
            make_candidate("b", 2, 0.8, 100),
        ];
        let alloc = budget.allocate(&candidates);
        assert_eq!(alloc.selected.len(), 2);
        assert_eq!(alloc.total_tokens, 200);
    }

    #[test]
    fn test_layer_overflow_truncates() {
        let budget = TokenBudget::new();
        // L3 quota is 300 tokens. Each item is 100. 4 items → only 3 fit.
        let candidates: Vec<BudgetCandidate> = (0..4)
            .map(|i| make_candidate(&format!("l3-{}", i), 3, 1.0 - i as f64 * 0.1, 100))
            .collect();
        let alloc = budget.allocate(&candidates);
        assert_eq!(alloc.selected.len(), 3);
        assert_eq!(alloc.total_tokens, 300);
    }

    #[test]
    fn test_redistribution_to_lower_layers() {
        let budget = TokenBudget::new();
        // L3 has nothing → its 300 tokens should flow to L2 (600+300=900).
        let candidates: Vec<BudgetCandidate> = (0..10)
            .map(|i| make_candidate(&format!("l2-{}", i), 2, 1.0 - i as f64 * 0.05, 100))
            .collect();
        let alloc = budget.allocate(&candidates);
        // L2 can now use up to 900 tokens → 9 items × 100 = 900
        assert_eq!(alloc.selected.len(), 9);
        assert_eq!(alloc.total_tokens, 900);
    }

    #[test]
    fn test_total_never_exceeds_budget() {
        let budget = TokenBudget::new();
        // 200 candidates across all layers, each 50 tokens = 10000 total.
        let candidates: Vec<BudgetCandidate> = (0..200)
            .map(|i| make_candidate(&format!("c-{}", i), (i % 4) as u8, 0.5, 50))
            .collect();
        let alloc = budget.allocate(&candidates);
        assert!(
            alloc.total_tokens <= budget.budget,
            "total_tokens {} exceeds budget {}",
            alloc.total_tokens,
            budget.budget
        );
    }

    #[test]
    fn test_high_score_items_prioritized() {
        let budget = TokenBudget::new();
        // Fill L3 so it doesn't redistribute its quota downward.
        let l3_filler: Vec<BudgetCandidate> = (0..3)
            .map(|i| make_candidate(&format!("l3-{}", i), 3, 0.9, 100))
            .collect();
        // L2 quota = 600. Three items: 300+200+200=700 > 600.
        let mut candidates = l3_filler;
        candidates.push(make_candidate("high", 2, 1.0, 300));
        candidates.push(make_candidate("mid", 2, 0.5, 200));
        candidates.push(make_candidate("low", 2, 0.1, 200));
        let alloc = budget.allocate(&candidates);
        let ids: Vec<&str> = alloc.selected.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"high"), "high-score item must be included");
        assert!(ids.contains(&"mid"), "mid-score item must be included");
        assert!(!ids.contains(&"low"), "low-score item should be excluded when quota is tight");
    }

    #[test]
    fn test_custom_budget() {
        let budget = TokenBudget::with_budget(1000);
        let q = budget.quotas();
        assert_eq!(q[0], 100); // L0: 10%
        assert_eq!(q[1], 300); // L1: 30%
        assert_eq!(q[2], 400); // L2: 40%
        assert_eq!(q[3], 200); // L3: 20%
    }

    #[test]
    fn test_all_layers_represented() {
        let budget = TokenBudget::new();
        let candidates = vec![
            make_candidate("l0", 0, 0.9, 50),
            make_candidate("l1", 1, 0.9, 50),
            make_candidate("l2", 2, 0.9, 50),
            make_candidate("l3", 3, 0.9, 50),
        ];
        let alloc = budget.allocate(&candidates);
        assert_eq!(alloc.selected.len(), 4);
        // All layers should have some usage
        for (_, used, _) in &alloc.layer_usage {
            assert!(*used > 0, "layer should have non-zero usage");
        }
    }
}
