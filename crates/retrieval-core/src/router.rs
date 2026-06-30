//! Metacognitive retrieval router with automatic drill-down
//!
//! After an initial retrieval pass, the router evaluates whether the
//! results are "confident enough" or whether the system should drill
//! down to deeper layers (L1/L0) for better evidence.

/// Metacognitive retrieval router.
///
/// Decides whether to drill down from high-level results (L2/L3) to
/// lower-level evidence (L1/L0) based on result quality heuristics.
pub struct MetaRouter {
    /// Minimum confidence score (0.0–1.0) below which drill-down is triggered.
    pub confidence_threshold: f64,
}

impl Default for MetaRouter {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.65,
        }
    }
}

impl MetaRouter {
    /// Create a router with the default confidence threshold (0.65).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a router with a custom confidence threshold.
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            confidence_threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Decide whether to drill down given a confidence score.
    ///
    /// Returns `true` when confidence is below the threshold.
    pub fn should_drill_down(&self, confidence: f64) -> bool {
        confidence < self.confidence_threshold
    }

    /// Evaluate confidence from retrieval result metadata.
    ///
    /// Simple heuristic:
    /// - If no results, confidence = 0.0 (always drill down).
    /// - Otherwise, confidence is proportional to the top result's score,
    ///   penalised when the result set is thin (< 3 items).
    pub fn evaluate_confidence(&self, results_count: usize, top_score: f64) -> f64 {
        if results_count == 0 {
            return 0.0;
        }

        // Thin result sets reduce confidence
        let thinness_penalty = if results_count < 3 { 0.7 } else { 1.0 };

        // Clamp top_score to [0, 1] for confidence
        let normalised_score = top_score.clamp(0.0, 1.0);

        (normalised_score * thinness_penalty).clamp(0.0, 1.0)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_confidence_no_drill() {
        let router = MetaRouter::new();
        let confidence = router.evaluate_confidence(10, 0.9);
        assert!(!router.should_drill_down(confidence));
    }

    #[test]
    fn test_low_confidence_drill() {
        let router = MetaRouter::new();
        let confidence = router.evaluate_confidence(1, 0.3);
        assert!(router.should_drill_down(confidence));
    }

    #[test]
    fn test_empty_results_always_drill() {
        let router = MetaRouter::new();
        let confidence = router.evaluate_confidence(0, 0.0);
        assert!(router.should_drill_down(confidence));
        assert!((confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thin_results_reduce_confidence() {
        let router = MetaRouter::new();
        // Same top_score, but 1 result vs 10 results
        let thin = router.evaluate_confidence(1, 0.9);
        let thick = router.evaluate_confidence(10, 0.9);
        assert!(thin < thick);
    }
}
