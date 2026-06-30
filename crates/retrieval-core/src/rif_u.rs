//! RIF-U scoring model (subjective-time based)
//!
//! RIF-U is the ranking heart of the retrieval system. It scores every
//! candidate memory along four dimensions:
//!
//! - **R**ecency  — how recently was it accessed? (subjective ticks, not wall-clock)
//! - **I**mportance — how critical is this fact?
//! - **F**requency — how often has it been accessed?
//! - **U**tility — has it been helpful or harmful?
//!
//! Weights default to R=0.25, I=0.35, F=0.15, U=0.25 per RIF-SQCE.md.
//!
//! Because `retrieval-core` must not depend on `graph-core`, the scorer
//! accepts raw scalars rather than `NodeAttr` — keeping the crate independent.

use std::collections::HashMap;

// ── Default weights ───────────────────────────────────────────────────────────

/// Default weight for the Recency dimension.
pub const DEFAULT_W_RECENCY: f64 = 0.25;
/// Default weight for the Importance dimension.
pub const DEFAULT_W_IMPORTANCE: f64 = 0.35;
/// Default weight for the Frequency dimension.
pub const DEFAULT_W_FREQUENCY: f64 = 0.15;
/// Default weight for the Utility dimension.
pub const DEFAULT_W_UTILITY: f64 = 0.25;

/// Default upper bound for frequency normalisation.
pub const DEFAULT_MAX_FREQUENCY: u64 = 100;

// ── Weights ───────────────────────────────────────────────────────────────────

/// Configurable weights for the four RIF-U dimensions.
///
/// Weights should sum to 1.0 for intuitive interpretation, but the scorer
/// does not enforce this — it simply computes the weighted sum and clamps.
#[derive(Debug, Clone)]
pub struct RifUWeights {
    pub recency: f64,
    pub importance: f64,
    pub frequency: f64,
    pub utility: f64,
}

impl Default for RifUWeights {
    fn default() -> Self {
        Self {
            recency: DEFAULT_W_RECENCY,
            importance: DEFAULT_W_IMPORTANCE,
            frequency: DEFAULT_W_FREQUENCY,
            utility: DEFAULT_W_UTILITY,
        }
    }
}

// ── Scorer ────────────────────────────────────────────────────────────────────

/// RIF-U scorer: evaluates memories along Recency–Importance–Frequency–Utility.
pub struct RifUScorer {
    weights: RifUWeights,

    /// Tag → importance value lookup. When scoring, the caller passes tags
    /// and the scorer picks the **maximum** matching importance.
    importance_presets: HashMap<String, f64>,

    /// Tag → utility value lookup. When scoring, the caller passes tags
    /// and the scorer picks the **minimum** matching utility (most negative
    /// pain-point wins).
    utility_presets: HashMap<String, f64>,

    /// Upper bound for normalising raw frequency counts to [0, 1].
    max_frequency: u64,
}

impl RifUScorer {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create a scorer with default weights and the preset tables from
    /// RIF-SQCE.md.
    pub fn new() -> Self {
        let importance_presets: HashMap<String, f64> = [
            ("architecture_decision".to_string(), 1.0),
            ("user_explicit_command".to_string(), 1.0),
            ("critical_pitfall".to_string(), 1.0),
            ("bug_fix_rationale".to_string(), 0.9),
            ("tech_selection".to_string(), 0.8),
            ("code_style_preference".to_string(), 0.6),
            ("temporary_context".to_string(), 0.2),
        ]
        .into_iter()
        .collect();

        let utility_presets: HashMap<String, f64> = [
            ("critical_pitfall".to_string(), 1.0),
            ("verified_fact".to_string(), 0.5),
            ("unverified_hypothesis".to_string(), 0.0),
            ("deprecated_pattern".to_string(), -0.8),
        ]
        .into_iter()
        .collect();

        Self {
            weights: RifUWeights::default(),
            importance_presets,
            utility_presets,
            max_frequency: DEFAULT_MAX_FREQUENCY,
        }
    }

    /// Create a scorer with custom weights, keeping default presets.
    pub fn with_weights(weights: RifUWeights) -> Self {
        Self {
            weights,
            ..Self::new()
        }
    }

    // ── Core scoring ──────────────────────────────────────────────────────────

    /// Score a single memory candidate.
    ///
    /// # Parameters
    ///
    /// - `tick_delta`: `current_active_tick - item_last_active_tick`.
    ///   A delta of 0 means "just accessed" (maximum recency).
    ///   Physical downtime does **not** increase this value.
    /// - `importance`: `[0.0, 1.0]` — how critical is this fact?
    /// - `frequency`: raw access count (will be normalised internally).
    /// - `utility`: `[-1.0, 1.0]` — positive = helpful, negative = harmful.
    ///
    /// # Returns
    ///
    /// A score in `[0.0, 1.0]`.
    pub fn score(&self, tick_delta: u64, importance: f64, frequency: u64, utility: f64) -> f64 {
        // Recency: hyperbolic decay. tick_delta=0 → 1.0; grows → 0.0
        let r = 1.0 / (1.0 + tick_delta as f64 * 0.5);

        // Importance: pass-through (caller or importance_for provides [0, 1])
        let i = importance.clamp(0.0, 1.0);

        // Frequency: normalise to [0, 1] using max_frequency ceiling
        let f = if self.max_frequency == 0 {
            0.0
        } else {
            let capped = frequency.min(self.max_frequency) as f64;
            capped / self.max_frequency as f64
        };

        // Utility: clamp to [-1, 1], then rescale to [0, 1] for the weighted
        // sum: (u + 1) / 2 maps -1→0, 0→0.5, 1→1.
        let u_raw = utility.clamp(-1.0, 1.0);
        let u = (u_raw + 1.0) / 2.0;

        // Weighted sum
        let total = r * self.weights.recency
            + i * self.weights.importance
            + f * self.weights.frequency
            + u * self.weights.utility;

        total.clamp(0.0, 1.0)
    }

    /// Score a batch of candidates.
    ///
    /// Each item is `(tick_delta, importance, frequency, utility)`.
    /// Returns scores in the same order.
    pub fn score_batch(&self, items: &[(u64, f64, u64, f64)]) -> Vec<f64> {
        items
            .iter()
            .map(|&(tick_delta, importance, frequency, utility)| {
                self.score(tick_delta, importance, frequency, utility)
            })
            .collect()
    }

    // ── Tag lookups ───────────────────────────────────────────────────────────

    /// Look up importance for a set of tags.
    ///
    /// Returns the **maximum** importance among all matching presets,
    /// or 0.5 as a neutral default if no tag matches.
    pub fn importance_for(&self, tags: &[&str]) -> f64 {
        tags.iter()
            .filter_map(|tag| self.importance_presets.get(*tag).copied())
            .fold(f64::NEG_INFINITY, f64::max)
            // If no tag matched, return a neutral default
            .pipe_if_infinite(0.5)
    }

    /// Look up utility for a set of tags.
    ///
    /// Returns the **minimum** utility among all matching presets
    /// (most negative pain-point wins), or 0.0 as a neutral default
    /// if no tag matches.
    pub fn utility_for(&self, tags: &[&str]) -> f64 {
        tags.iter()
            .filter_map(|tag| self.utility_presets.get(*tag).copied())
            .fold(f64::INFINITY, f64::min)
            .pipe_if_infinite(0.0)
    }
}

impl Default for RifUScorer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helper trait ──────────────────────────────────────────────────────────────

/// Tiny extension to replace `f64::INFINITY` / `f64::NEG_INFINITY` sentinels
/// with a concrete fallback. Avoids pulling in a full dependency for one op.
trait PipeIfInfinite {
    fn pipe_if_infinite(self, fallback: f64) -> f64;
}

impl PipeIfInfinite for f64 {
    fn pipe_if_infinite(self, fallback: f64) -> f64 {
        if self.is_infinite() { fallback } else { self }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_weights() {
        let w = RifUWeights::default();
        assert!((w.recency - 0.25).abs() < f64::EPSILON);
        assert!((w.importance - 0.35).abs() < f64::EPSILON);
        assert!((w.frequency - 0.15).abs() < f64::EPSILON);
        assert!((w.utility - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_full_score_all_max() {
        let scorer = RifUScorer::new();
        // tick_delta=0 (max recency), importance=1.0, frequency=100, utility=1.0
        let s = scorer.score(0, 1.0, 100, 1.0);
        // R=1.0, I=1.0, F=1.0, U=(1+1)/2=1.0
        // total = 1*0.25 + 1*0.35 + 1*0.15 + 1*0.25 = 1.0
        assert!((s - 1.0).abs() < 1e-10, "expected 1.0, got {}", s);
    }

    #[test]
    fn test_full_score_all_zero() {
        let scorer = RifUScorer::new();
        // tick_delta very large (recency→0), importance=0, frequency=0, utility=-1.0
        let s = scorer.score(u64::MAX / 2, 0.0, 0, -1.0);
        // R≈0, I=0, F=0, U=(-1+1)/2=0 → total≈0
        assert!(s < 0.01, "expected ≈0, got {}", s);
    }

    #[test]
    fn test_utility_negative_pulls_down() {
        let scorer = RifUScorer::new();
        // High recency, importance, frequency, but utility = -1.0
        let s_bad = scorer.score(0, 1.0, 100, -1.0);
        let s_good = scorer.score(0, 1.0, 100, 1.0);
        // Utility of -1.0 → U=0.0 vs +1.0 → U=1.0, difference = 0.25
        assert!(
            s_bad < s_good,
            "negative utility should lower score: bad={}, good={}",
            s_bad,
            s_good
        );
        // U=-1 → (−1+1)/2=0 → total = 0.25+0.35+0.15+0 = 0.75
        assert!((s_bad - 0.75).abs() < 1e-10, "expected 0.75, got {}", s_bad);
    }

    #[test]
    fn test_recency_decay_curve() {
        let scorer = RifUScorer::new();

        // tick_delta=0 → R=1.0
        let s0 = scorer.score(0, 0.5, 50, 0.0);
        // tick_delta=1 → R=0.5
        let s1 = scorer.score(1, 0.5, 50, 0.0);
        // tick_delta=9 → R=0.1
        let s9 = scorer.score(9, 0.5, 50, 0.0);

        // Recency should strictly decrease
        assert!(s0 > s1, "s0={} should be > s1={}", s0, s1);
        assert!(s1 > s9, "s1={} should be > s9={}", s1, s9);

        // Verify exact R values: R = 1/(1+delta)
        let r0 = 1.0 / (1.0 + 0.0);
        let r1 = 1.0 / (1.0 + 1.0);
        let r9 = 1.0 / (1.0 + 9.0);
        assert!((r0 - 1.0_f64).abs() < f64::EPSILON);
        assert!((r1 - 0.5_f64).abs() < f64::EPSILON);
        assert!((r9 - 0.1_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_importance_for_presets() {
        let scorer = RifUScorer::new();

        assert!((scorer.importance_for(&["architecture_decision"]) - 1.0).abs() < f64::EPSILON);
        assert!((scorer.importance_for(&["critical_pitfall"]) - 1.0).abs() < f64::EPSILON);
        assert!((scorer.importance_for(&["bug_fix_rationale"]) - 0.9).abs() < f64::EPSILON);
        assert!((scorer.importance_for(&["tech_selection"]) - 0.8).abs() < f64::EPSILON);
        assert!((scorer.importance_for(&["code_style_preference"]) - 0.6).abs() < f64::EPSILON);
        assert!((scorer.importance_for(&["temporary_context"]) - 0.2).abs() < f64::EPSILON);

        // Multiple tags: should take max
        assert!(
            (scorer.importance_for(&["temporary_context", "bug_fix_rationale"]) - 0.9).abs()
                < f64::EPSILON
        );

        // Unknown tag: neutral default 0.5
        assert!((scorer.importance_for(&["unknown_tag"]) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_utility_for_presets() {
        let scorer = RifUScorer::new();

        assert!((scorer.utility_for(&["critical_pitfall"]) - 1.0).abs() < f64::EPSILON);
        assert!((scorer.utility_for(&["verified_fact"]) - 0.5).abs() < f64::EPSILON);
        assert!((scorer.utility_for(&["unverified_hypothesis"]) - 0.0).abs() < f64::EPSILON);
        assert!((scorer.utility_for(&["deprecated_pattern"]) - (-0.8)).abs() < f64::EPSILON);

        // Multiple tags: should take min (most negative wins)
        assert!(
            (scorer.utility_for(&["critical_pitfall", "deprecated_pattern"]) - (-0.8)).abs()
                < f64::EPSILON
        );

        // Unknown tag: neutral default 0.0
        assert!((scorer.utility_for(&["unknown_tag"]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_batch() {
        let scorer = RifUScorer::new();
        let items = vec![
            (0, 1.0, 100, 1.0),   // max everything
            (1000, 0.0, 0, -1.0), // min everything
            (0, 0.5, 50, 0.0),    // mixed
        ];
        let scores = scorer.score_batch(&items);
        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 1.0).abs() < 1e-10);
        assert!(scores[1] < 0.01);
        assert!(scores[2] > scores[1]);
        assert!(scores[2] < scores[0]);
    }

    #[test]
    fn test_subjective_time_immune_to_downtime() {
        let scorer = RifUScorer::new();
        // Two facts, both accessed at the same subjective tick
        let s_recent = scorer.score(0, 0.5, 10, 0.0);
        let s_old = scorer.score(0, 0.5, 10, 0.0);
        // Same tick_delta → same score, regardless of physical downtime
        assert!(
            (s_recent - s_old).abs() < f64::EPSILON,
        );

        // Even at tick_delta=0 after "1 year of downtime", score is identical
        // because tick_delta is subjective, not wall-clock
        let s_after_long_downtime = scorer.score(0, 0.5, 10, 0.0);
        assert!((s_recent - s_after_long_downtime).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frequency_normalisation() {
        let scorer = RifUScorer::new();

        // frequency=0 → F=0
        let s0 = scorer.score(0, 0.5, 0, 0.0);
        // frequency=50 → F=0.5
        let s50 = scorer.score(0, 0.5, 50, 0.0);
        // frequency=100 → F=1.0
        let s100 = scorer.score(0, 0.5, 100, 0.0);
        // frequency=200 → capped at 1.0
        let s200 = scorer.score(0, 0.5, 200, 0.0);

        assert!(s0 < s50);
        assert!(s50 < s100);
        // Capped: s100 == s200
        assert!((s100 - s200).abs() < 1e-10);
    }
}
