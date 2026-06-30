//! Cognitive DNA genome storage
//!
//! Encodes the system's tunable "thought strategy" parameters into a
//! single [`CognitiveDna`] struct that can be mutated, crossed-over,
//! and evaluated by the genetic-algorithm sandbox (Epic 10).

use serde::{Deserialize, Serialize};

/// The complete set of tunable cognitive parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveDna {
    /// RIF-U weights: (recency, importance, frequency, utility).
    pub rif_weights: (f64, f64, f64, f64),

    /// Metacognitive router confidence threshold for drill-down.
    pub confidence_threshold: f64,

    /// TDA Betti-number threshold that triggers a "phase transition" alert.
    pub betti_threshold: u32,

    /// Minimum call count before an inference path is promoted to a cognitive macro.
    pub macro_compilation_threshold: u64,

    /// Assimilation-failure-rate threshold that triggers a paradigm-shift crisis.
    pub failure_rate_threshold: f64,

    /// Genetic fitness score assigned by the sandbox evaluation.
    pub fitness_score: f64,
}

impl Default for CognitiveDna {
    fn default() -> Self {
        Self {
            rif_weights: (0.25, 0.35, 0.15, 0.25),
            confidence_threshold: 0.65,
            betti_threshold: 5,
            macro_compilation_threshold: 20,
            failure_rate_threshold: 0.3,
            fitness_score: 0.0,
        }
    }
}

impl CognitiveDna {
    /// Create a DNA instance with default (documented) parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a random mutation to one or more parameters.
    ///
    /// Each parameter is perturbed by a small random offset (±max_delta).
    /// All values are clamped to their valid ranges.
    pub fn mutate(&self, max_delta: f64) -> Self {
        let delta = |base: f64, lo: f64, hi: f64| -> f64 {
            let perturbation = (pseudorandom(base) - 0.5) * 2.0 * max_delta;
            (base + perturbation).clamp(lo, hi)
        };

        Self {
            rif_weights: (
                delta(self.rif_weights.0, 0.0, 1.0),
                delta(self.rif_weights.1, 0.0, 1.0),
                delta(self.rif_weights.2, 0.0, 1.0),
                delta(self.rif_weights.3, 0.0, 1.0),
            ),
            confidence_threshold: delta(self.confidence_threshold, 0.0, 1.0),
            betti_threshold: self.betti_threshold, // integer, skip mutation for simplicity
            macro_compilation_threshold: self.macro_compilation_threshold,
            failure_rate_threshold: delta(self.failure_rate_threshold, 0.0, 1.0),
            fitness_score: 0.0, // reset fitness after mutation
        }
    }

    /// Two-point crossover with another DNA strand.
    pub fn crossover(&self, other: &Self, crossover_point: usize) -> Self {
        // Simple field-level crossover based on an index
        let pick = |idx: usize, a: f64, b: f64| -> f64 {
            if idx < crossover_point { a } else { b }
        };

        Self {
            rif_weights: (
                pick(0, self.rif_weights.0, other.rif_weights.0),
                pick(1, self.rif_weights.1, other.rif_weights.1),
                pick(2, self.rif_weights.2, other.rif_weights.2),
                pick(3, self.rif_weights.3, other.rif_weights.3),
            ),
            confidence_threshold: pick(4, self.confidence_threshold, other.confidence_threshold),
            betti_threshold: if crossover_point <= 5 {
                self.betti_threshold
            } else {
                other.betti_threshold
            },
            macro_compilation_threshold: if crossover_point <= 6 {
                self.macro_compilation_threshold
            } else {
                other.macro_compilation_threshold
            },
            failure_rate_threshold: pick(7, self.failure_rate_threshold, other.failure_rate_threshold),
            fitness_score: 0.0,
        }
    }
}

/// Deterministic pseudo-random function (no external RNG dependency).
/// Returns a value in [0.0, 1.0) based on a simple hash of the input.
fn pseudorandom(seed: f64) -> f64 {
    let bits = seed.to_bits().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (bits >> 33) as f64 / (1u64 << 31) as f64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let dna = CognitiveDna::new();
        assert!((dna.rif_weights.0 - 0.25).abs() < f64::EPSILON);
        assert!((dna.rif_weights.1 - 0.35).abs() < f64::EPSILON);
        assert!((dna.confidence_threshold - 0.65).abs() < f64::EPSILON);
        assert_eq!(dna.betti_threshold, 5);
        assert_eq!(dna.macro_compilation_threshold, 20);
    }

    #[test]
    fn test_mutate_stays_in_range() {
        let dna = CognitiveDna::new();
        for _ in 0..100 {
            let mutated = dna.mutate(0.3);
            assert!((0.0..=1.0).contains(&mutated.rif_weights.0));
            assert!((0.0..=1.0).contains(&mutated.rif_weights.1));
            assert!((0.0..=1.0).contains(&mutated.confidence_threshold));
            assert!((0.0..=1.0).contains(&mutated.failure_rate_threshold));
            assert!((mutated.fitness_score - 0.0).abs() < f64::EPSILON);
        }
    }
}
