//! Cognitive macro compilation cache
//!
//! When a particular inference pattern is called frequently and succeeds
//! often, it gets "compiled" into a [`CognitiveMacro`] — a cached shortcut
//! that bypasses the full retrieval pipeline on future similar inputs.
//!
//! This implements the "cognitive myelination" mechanism from Epic 10.

use std::collections::HashMap;

/// A compiled cognitive macro: a cached inference shortcut.
#[derive(Debug, Clone)]
pub struct CognitiveMacro {
    /// Input pattern / feature description that triggers this macro.
    pub pattern: String,
    /// Cached result / output template.
    pub result: String,
    /// Number of times this macro has been triggered.
    pub hit_count: u64,
    /// Rolling success rate in [0.0, 1.0].
    pub success_rate: f64,
    /// The active-tick timestamp when this macro was compiled.
    pub compiled_at_tick: u64,
}

/// In-memory cache of compiled cognitive macros.
pub struct MacroCache {
    macros: HashMap<String, CognitiveMacro>,
    /// Minimum hit count before a pattern is promoted to a "fast path" macro.
    promotion_threshold: u64,
}

impl MacroCache {
    /// Create an empty macro cache with the given promotion threshold.
    pub fn new(promotion_threshold: u64) -> Self {
        Self {
            macros: HashMap::new(),
            promotion_threshold,
        }
    }

    /// Look up a macro by its pattern key.
    pub fn get(&self, pattern: &str) -> Option<&CognitiveMacro> {
        self.macros.get(pattern)
    }

    /// Insert or update a macro in the cache.
    pub fn insert(&mut self, mac: CognitiveMacro) {
        self.macros.insert(mac.pattern.clone(), mac);
    }

    /// Record a hit for a pattern (increment hit_count, update success_rate).
    pub fn record_hit(&mut self, pattern: &str, success: bool) {
        if let Some(mac) = self.macros.get_mut(pattern) {
            mac.hit_count += 1;
            // Exponential moving average
            let alpha = 0.1;
            let new_val = if success { 1.0 } else { 0.0 };
            mac.success_rate = (1.0 - alpha) * mac.success_rate + alpha * new_val;
        }
    }

    /// Return all macros whose hit_count meets or exceeds the promotion threshold.
    pub fn promoted(&self) -> Vec<&CognitiveMacro> {
        self.macros
            .values()
            .filter(|m| m.hit_count >= self.promotion_threshold)
            .collect()
    }

    /// Number of macros in the cache.
    pub fn len(&self) -> usize {
        self.macros.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.macros.is_empty()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut cache = MacroCache::new(5);
        cache.insert(CognitiveMacro {
            pattern: "bug_pattern_A".to_string(),
            result: "fix_A".to_string(),
            hit_count: 1,
            success_rate: 0.5,
            compiled_at_tick: 42,
        });

        let mac = cache.get("bug_pattern_A").unwrap();
        assert_eq!(mac.result, "fix_A");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_promotion_threshold() {
        let mut cache = MacroCache::new(3);
        cache.insert(CognitiveMacro {
            pattern: "low_hits".to_string(),
            result: "r".to_string(),
            hit_count: 1,
            success_rate: 0.8,
            compiled_at_tick: 0,
        });
        cache.insert(CognitiveMacro {
            pattern: "high_hits".to_string(),
            result: "r".to_string(),
            hit_count: 5,
            success_rate: 0.9,
            compiled_at_tick: 0,
        });

        let promoted = cache.promoted();
        assert_eq!(promoted.len(), 1);
        assert_eq!(promoted[0].pattern, "high_hits");
    }

    #[test]
    fn test_record_hit_updates_rate() {
        let mut cache = MacroCache::new(1);
        cache.insert(CognitiveMacro {
            pattern: "p".to_string(),
            result: "r".to_string(),
            hit_count: 0,
            success_rate: 0.5,
            compiled_at_tick: 0,
        });

        cache.record_hit("p", true);
        let mac = cache.get("p").unwrap();
        assert_eq!(mac.hit_count, 1);
        // success_rate should increase (alpha=0.1: 0.9*0.5 + 0.1*1.0 = 0.55)
        assert!((mac.success_rate - 0.55).abs() < 1e-10);
    }
}
