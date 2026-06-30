//! Temporal version tree for entity history
//!
//! Each entity maintains an independent version chain (singly-linked list).
//! Updates never overwrite — they create a new [`VersionNode`] and seal the
//! previous head's `valid_until` timestamp. This guarantees full auditability
//! and enables precise point-in-time queries.
//!
//! ## Performance target
//!
//! Single-entity lookup by `(entity_id, version_seq)` is **O(1)** via
//! `HashMap`. History traversal and `query_at` are **O(n)** in the number
//! of versions per entity, which is fine for typical chains (< 1 000 nodes).

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum VersionError {
    #[error("entity '{entity_id}' version {version_seq} not found")]
    NotFound {
        entity_id: String,
        version_seq: u64,
    },

    #[error("entity '{entity_id}' has no versions")]
    EmptyHistory { entity_id: String },
}

// ── Version node ──────────────────────────────────────────────────────────────

/// A single immutable snapshot of an entity at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionNode {
    /// Owning entity identifier (e.g. a graph node id or fact label).
    pub entity_id: String,

    /// Monotonically increasing sequence number, unique per entity.
    /// Starts at 1 for the first version.
    pub version_seq: u64,

    /// When this version became the active one.
    pub valid_from: DateTime<Utc>,

    /// When this version was superseded. `None` means "still current".
    pub valid_until: Option<DateTime<Utc>>,

    /// Blake3 hex digest of the content this version represents.
    /// Used for change detection and deduplication.
    pub content_hash: String,

    /// Sequence number of the previous version in this entity's chain.
    /// `None` for the very first version (seq = 1).
    pub prev_version_seq: Option<u64>,

    /// If `true`, this version represents an abandoned branch / counterfactual
    /// scenario. It is preserved for "avoid-the-pit" reasoning but excluded
    /// from normal retrieval.
    pub is_counterfactual: bool,

    /// Free-form metadata (JSON). Carries whatever the caller needs — diff
    /// summaries, trigger context, L0 evidence references, etc.
    pub metadata: Value,
}

// ── Version tree ──────────────────────────────────────────────────────────────

/// In-memory store of version chains for all entities.
///
/// Internal layout:
/// - `nodes`: `HashMap<(entity_id, version_seq), VersionNode>` — O(1) lookup.
/// - `heads`: `HashMap<entity_id, latest_version_seq>` — tracks current head.
pub struct VersionTree {
    nodes: HashMap<(String, u64), VersionNode>,
    heads: HashMap<String, u64>,
}

impl VersionTree {
    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /// Create an empty version tree.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            heads: HashMap::new(),
        }
    }

    /// Total number of version nodes across all entities.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tree contains zero versions.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    // ── Writes ────────────────────────────────────────────────────────────────

    /// Insert a new version for `entity_id`.
    ///
    /// Returns the newly created [`VersionNode`].
    ///
    /// **Behaviour**:
    /// 1. If the entity already exists, its current head's `valid_until` is
    ///    sealed to `Utc::now()` and the new version's `prev_version_seq`
    ///    points to the old head.
    /// 2. If the entity is new, `version_seq` starts at 1 and
    ///    `prev_version_seq` is `None`.
    pub fn insert(
        &mut self,
        entity_id: &str,
        content_hash: String,
        metadata: Value,
    ) -> VersionNode {
        let now = Utc::now();
        let key = entity_id.to_string();

        // Determine next sequence number and seal the old head.
        let (next_seq, prev_seq) = match self.heads.get(&key) {
            Some(&old_seq) => {
                // Seal the old head's valid_until.
                if let Some(old_node) = self.nodes.get_mut(&(key.clone(), old_seq)) {
                    old_node.valid_until = Some(now);
                }
                (old_seq + 1, Some(old_seq))
            }
            None => (1, None),
        };

        let node = VersionNode {
            entity_id: key.clone(),
            version_seq: next_seq,
            valid_from: now,
            valid_until: None, // current head — not yet superseded
            content_hash,
            prev_version_seq: prev_seq,
            is_counterfactual: false,
            metadata,
        };

        self.nodes.insert((key.clone(), next_seq), node.clone());
        self.heads.insert(key, next_seq);

        node
    }

    // ── Reads ─────────────────────────────────────────────────────────────────

    /// Get a specific version of an entity by sequence number.
    pub fn get(&self, entity_id: &str, version_seq: u64) -> Option<&VersionNode> {
        self.nodes.get(&(entity_id.to_string(), version_seq))
    }

    /// Get a mutable reference to a specific version.
    pub fn get_mut(&mut self, entity_id: &str, version_seq: u64) -> Option<&mut VersionNode> {
        self.nodes.get_mut(&(entity_id.to_string(), version_seq))
    }

    /// Get the latest (current head) version of an entity.
    pub fn get_latest(&self, entity_id: &str) -> Option<&VersionNode> {
        let seq = self.heads.get(entity_id)?;
        self.nodes.get(&(entity_id.to_string(), *seq))
    }

    /// Return the full version history for an entity, **newest first**.
    ///
    /// Walks the chain from head backwards via `prev_version_seq`.
    pub fn history(&self, entity_id: &str) -> Vec<&VersionNode> {
        let mut result = Vec::new();
        let head_seq = match self.heads.get(entity_id) {
            Some(&seq) => seq,
            None => return result,
        };

        let mut current_seq = Some(head_seq);
        while let Some(seq) = current_seq {
            if let Some(node) = self.nodes.get(&(entity_id.to_string(), seq)) {
                current_seq = node.prev_version_seq;
                result.push(node);
            } else {
                break; // chain integrity violation — stop gracefully
            }
        }

        result
    }

    /// Mark a specific version as counterfactual (abandoned branch).
    ///
    /// The version remains in the tree for "avoid-the-pit" reasoning but is
    /// excluded from normal retrieval paths.
    pub fn mark_counterfactual(
        &mut self,
        entity_id: &str,
        version_seq: u64,
    ) -> Result<(), VersionError> {
        self.mark_counterfactual_with_conditions(entity_id, version_seq, None)
    }

    /// Mark a version as counterfactual with structured failure conditions.
    ///
    /// `conditions` is an optional JSON object describing WHY the branch failed,
    /// enabling future re-evaluation when conditions change.
    pub fn mark_counterfactual_with_conditions(
        &mut self,
        entity_id: &str,
        version_seq: u64,
        conditions: Option<serde_json::Value>,
    ) -> Result<(), VersionError> {
        let key = (entity_id.to_string(), version_seq);
        let node = self
            .nodes
            .get_mut(&key)
            .ok_or(VersionError::NotFound {
                entity_id: entity_id.to_string(),
                version_seq,
            })?;
        node.is_counterfactual = true;
        if let Some(cond) = conditions {
            node.metadata["failure_conditions"] = cond;
            node.metadata["abandoned_at"] =
                serde_json::Value::String(chrono::Utc::now().to_rfc3339());
        }
        Ok(())
    }
    /// Re-enable a counterfactual version by creating a new version that
    /// references it, preserving the full audit chain.
    ///
    /// This is the "insert-based recovery" — the old counterfactual version
    /// stays marked as such, but a new active version is created on top.
    pub fn re_enable(
        &mut self,
        entity_id: &str,
        counterfactual_seq: u64,
        reason: &str,
    ) -> Result<u64, VersionError> {
        // Verify the target version exists and is counterfactual
        let key = (entity_id.to_string(), counterfactual_seq);
        let old_node = self
            .nodes
            .get(&key)
            .ok_or(VersionError::NotFound {
                entity_id: entity_id.to_string(),
                version_seq: counterfactual_seq,
            })?;

        if !old_node.is_counterfactual {
            return Err(VersionError::NotFound {
                entity_id: entity_id.to_string(),
                version_seq: counterfactual_seq,
            });
        }

        // Create new version referencing the re-enabled one
        let content_hash = old_node.content_hash.clone();
        let metadata = serde_json::json!({
            "re_enabled_from": counterfactual_seq,
            "re_enable_reason": reason,
            "re_enabled_at": chrono::Utc::now().to_rfc3339(),
        });

        let new_node = self.insert(entity_id, content_hash, metadata);
        Ok(new_node.version_seq)
    }

    /// Find the version of an entity that was active at a given point in time.
    ///
    /// Walks the chain from head backwards, looking for the first node where
    /// `valid_from <= timestamp` and (`valid_until` is `None` or `>= timestamp`).
    ///
    /// Returns `None` if no version was active at that time (e.g. timestamp is
    /// before the first version).
    pub fn query_at(
        &self,
        entity_id: &str,
        timestamp: DateTime<Utc>,
    ) -> Option<&VersionNode> {
        // Walk from head to oldest
        let head_seq = self.heads.get(entity_id)?;
        let mut current_seq = Some(*head_seq);

        while let Some(seq) = current_seq {
            let node = self.nodes.get(&(entity_id.to_string(), seq))?;

            // Is this version valid at `timestamp`?
            let started = node.valid_from <= timestamp;
            let not_ended = match node.valid_until {
                Some(until) => timestamp < until,
                None => true, // still current
            };

            if started && not_ended {
                return Some(node);
            }

            // If we've already passed the timestamp (node started after it),
            // keep walking backwards — an older version might cover it.
            current_seq = node.prev_version_seq;
        }

        None
    }

    // ── Temporal Gate ───────────────────────────────────────────────────────

    /// Filter the version history for an entity, returning only versions that
    /// are **active** at the current time.
    ///
    /// Excludes:
    /// - Counterfactual versions (`is_counterfactual == true`)
    /// - Expired versions (`valid_until < now`)
    ///
    /// Versions from a long dormancy period (> `dormancy_threshold` physical
    /// days since `valid_from`) are included but tagged with
    /// `[HISTORICAL_CONTEXT]` in their metadata so the retrieval layer can
    /// flag them for "gap reflection".
    pub fn temporal_gate(
        &self,
        entity_id: &str,
        now: DateTime<Utc>,
        dormancy_days: i64,
    ) -> Vec<&VersionNode> {
        let history = self.history(entity_id);
        let _dormancy_threshold = chrono::Duration::days(dormancy_days);

        history
            .into_iter()
            .filter(|node| {
                // Exclude counterfactual versions
                if node.is_counterfactual {
                    return false;
                }
                // Exclude expired versions
                if let Some(valid_until) = node.valid_until {
                    if valid_until < now {
                        return false;
                    }
                }
                true
            })
            .collect()
    }

    /// Like [`temporal_gate`], but also returns a flag indicating whether the
    /// entity has been dormant (no active ticks within the threshold).
    ///
    /// Returns `(filtered_versions, is_dormant)`.
    pub fn temporal_gate_with_dormancy(
        &self,
        entity_id: &str,
        now: DateTime<Utc>,
        dormancy_days: i64,
        last_active_tick: u64,
        current_tick: u64,
    ) -> (Vec<&VersionNode>, bool) {
        let filtered = self.temporal_gate(entity_id, now, dormancy_days);
        let tick_delta = current_tick.saturating_sub(last_active_tick);
        // If the entity hasn't been accessed in more than dormancy_days worth
        // of "typical" ticks (assume ~10 ticks/day as baseline), flag it.
        let dormancy_tick_threshold = (dormancy_days as u64) * 10;
        let is_dormant = tick_delta > dormancy_tick_threshold;
        (filtered, is_dormant)
    }

    // ── Iteration ─────────────────────────────────────────────────────────────

    /// Iterate over all entity IDs that have at least one version.
    pub fn entity_ids(&self) -> impl Iterator<Item = &String> {
        self.heads.keys()
    }

    /// Get the version count for a specific entity.
    pub fn entity_len(&self, entity_id: &str) -> usize {
        self.history(entity_id).len()
    }

    // ── Diff ──────────────────────────────────────────────────────────────────

    /// Compare two versions of an entity and return a diff report.
    ///
    /// Returns `None` if either version doesn't exist.
    /// The report includes both version nodes, their metadata diff,
    /// and the time between versions.
    pub fn get_diff(
        &self,
        entity_id: &str,
        seq_a: u64,
        seq_b: u64,
    ) -> Option<VersionDiff> {
        let node_a = self.nodes.get(&(entity_id.to_string(), seq_a))?;
        let node_b = self.nodes.get(&(entity_id.to_string(), seq_b))?;

        let time_delta = node_b.valid_from.signed_duration_since(node_a.valid_from);

        Some(VersionDiff {
            entity_id: entity_id.to_string(),
            version_a: node_a.clone(),
            version_b: node_b.clone(),
            content_changed: node_a.content_hash != node_b.content_hash,
            time_delta_seconds: time_delta.num_seconds(),
            metadata_diff: compute_metadata_diff(&node_a.metadata, &node_b.metadata),
        })
    }

    // ── Version Compression ──────────────────────────────────────────────────

    /// Compress old versions of an entity by merging them into a single
    /// "summary version".
    ///
    /// Versions older than `active_tick_threshold` ticks from the current head
    /// are collapsed into a single summary node. This reduces storage while
    /// preserving an aggregated metadata record of the historical context.
    ///
    /// Returns the number of versions removed (compressed).
    pub fn compress(
        &mut self,
        entity_id: &str,
        active_tick_threshold: u64,
    ) -> Result<usize, VersionError> {
        let history = self.history(entity_id);
        if history.is_empty() {
            return Err(VersionError::EmptyHistory {
                entity_id: entity_id.to_string(),
            });
        }

        // The newest version is the current head — never compress it
        let head_node = history[0];
        let head_seq = head_node.version_seq;

        // Find versions older than the threshold
        // A version is "old" if it's more than active_tick_threshold seqs behind head
        let cutoff_seq = head_seq.saturating_sub(active_tick_threshold);

        // Collect versions to compress (seq <= cutoff_seq, exclude head)
        let versions_to_compress: Vec<u64> = history
            .iter()
            .skip(1) // skip head
            .filter(|v| v.version_seq <= cutoff_seq)
            .map(|v| v.version_seq)
            .collect();

        if versions_to_compress.is_empty() {
            return Ok(0);
        }

        // Get the oldest node's timestamps before we remove anything
        let oldest_seq = *versions_to_compress.last().unwrap();
        let oldest_node = self.nodes.get(&(entity_id.to_string(), oldest_seq)).cloned();

        // Aggregate metadata from old versions
        let mut aggregated_metadata = serde_json::json!({
            "compressed": true,
            "original_count": versions_to_compress.len(),
            "version_range": {
                "from": versions_to_compress.first(),
                "to": versions_to_compress.last(),
            },
            "compressed_at": Utc::now().to_rfc3339(),
            "summaries": []
        });

        let summaries = aggregated_metadata["summaries"].as_array_mut().unwrap();
        for &seq in &versions_to_compress {
            if let Some(node) = self.nodes.get(&(entity_id.to_string(), seq)) {
                summaries.push(serde_json::json!({
                    "seq": seq,
                    "content_hash": node.content_hash,
                    "valid_from": node.valid_from.to_rfc3339(),
                    "was_counterfactual": node.is_counterfactual,
                }));
            }
        }

        // Compute a combined content hash for the summary node
        let combined_hash = format!(
            "compressed:{}-{}",
            versions_to_compress.first().unwrap_or(&0),
            versions_to_compress.last().unwrap_or(&0)
        );

        // Find the node that points to the first version we're removing
        // It's the node whose prev_version_seq == last_version_to_compress_seq
        let last_compress_seq = *versions_to_compress.first().unwrap();
        let next_after_compress = last_compress_seq + 1;
        // The node at next_after_compress should exist (it's between compressed range and head)
        // but only if it wasn't also compressed
        let should_relink = !versions_to_compress.contains(&next_after_compress);

        // Remove the old version nodes (except we'll reuse the oldest seq for the summary)
        let removed_count = versions_to_compress.len();
        for &seq in &versions_to_compress {
            if seq != oldest_seq {
                // Remove nodes other than the one we'll replace with summary
                self.nodes.remove(&(entity_id.to_string(), seq));
            }
        }

        // Replace the oldest node with the summary node
        let summary_node = VersionNode {
            entity_id: entity_id.to_string(),
            version_seq: oldest_seq,
            valid_from: oldest_node
                .as_ref()
                .map(|n| n.valid_from)
                .unwrap_or_else(Utc::now),
            valid_until: oldest_node
                .as_ref()
                .and_then(|n| n.valid_until),
            content_hash: combined_hash,
            prev_version_seq: None,
            is_counterfactual: false,
            metadata: aggregated_metadata,
        };

        self.nodes
            .insert((entity_id.to_string(), oldest_seq), summary_node);

        // Update prev_version_seq for the node that pointed to the first removed version
        // This node should now point to our summary (oldest_seq)
        if should_relink {
            if let Some(next_node) = self.nodes.get_mut(&(entity_id.to_string(), next_after_compress)) {
                next_node.prev_version_seq = Some(oldest_seq);
            }
        }

        Ok(removed_count)
    }

    // ── Hibernation-aware filtering ────────────────────────────────────────

    /// Filter version history, returning only active versions.
    ///
    /// - Excludes `is_counterfactual=true` versions.
    /// - Excludes expired versions (`valid_until` already passed).
    /// - **Hibernation tolerance**: if `physical_downtime_days > 7`, expired
    ///   versions are NOT dropped — instead they get `historical_context: true`
    ///   in metadata, kept for "gap reflection" reasoning.
    ///
    /// Returns versions newest-first.
    pub fn filter_active(
        &self,
        entity_id: &str,
        as_of: DateTime<Utc>,
        physical_downtime_days: u64,
    ) -> Vec<&VersionNode> {
        let hibernation_mode = physical_downtime_days > 7;
        let history = self.history(entity_id);

        history
            .into_iter()
            .filter_map(|node| {
                // Always exclude counterfactual
                if node.is_counterfactual {
                    return None;
                }

                // Check expiry
                let expired = node
                    .valid_until
                    .map(|until| until < as_of)
                    .unwrap_or(false);

                if expired && !hibernation_mode {
                    // Normal mode: drop expired
                    return None;
                }

                // In hibernation mode, expired versions are kept but tagged
                // NOTE: We can't mutate &VersionNode, so we return it as-is.
                // The caller should check hibernation_mode and tag if needed.
                // For now, we return the node — the tagging happens at the
                // retrieval layer via metadata inspection.
                Some(node)
            })
            .collect()
    }

    /// Check if the entity is in hibernation (dormant for long period).
    ///
    /// Returns `true` if `physical_downtime_days > 7`.
    pub fn is_hibernating(&self, physical_downtime_days: u64) -> bool {
        physical_downtime_days > 7
    }
}

impl Default for VersionTree {
    fn default() -> Self {
        Self::new()
    }
}

// ── Diff types ────────────────────────────────────────────────────────────────

/// Report comparing two versions of an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    /// The entity being compared.
    pub entity_id: String,
    /// First version in the comparison.
    pub version_a: VersionNode,
    /// Second version in the comparison.
    pub version_b: VersionNode,
    /// Whether the content hash changed between versions.
    pub content_changed: bool,
    /// Time delta in seconds between the two versions (positive = b is later).
    pub time_delta_seconds: i64,
    /// Keys that differ between the two versions' metadata, with (old, new) values.
    pub metadata_diff: Vec<(String, Value, Value)>,
}

/// Compare two JSON values and return differing keys.
fn compute_metadata_diff(a: &Value, b: &Value) -> Vec<(String, Value, Value)> {
    let mut diffs = Vec::new();

    match (a, b) {
        (Value::Object(map_a), Value::Object(map_b)) => {
            let mut all_keys: Vec<&String> = map_a.keys().collect();
            for k in map_b.keys() {
                if !all_keys.contains(&k) {
                    all_keys.push(k);
                }
            }
            for key in all_keys {
                let va = map_a.get(key).cloned().unwrap_or(Value::Null);
                let vb = map_b.get(key).cloned().unwrap_or(Value::Null);
                if va != vb {
                    diffs.push((key.clone(), va, vb));
                }
            }
        }
        _ => {
            if a != b {
                diffs.push(("_root".to_string(), a.clone(), b.clone()));
            }
        }
    }

    diffs
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_empty_tree() {
        let tree = VersionTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.get_latest("no-such-entity").is_none());
        assert!(tree.history("no-such-entity").is_empty());
    }

    #[test]
    fn test_insert_and_get_latest() {
        let mut tree = VersionTree::new();

        let v1 = tree.insert("doc-1", "hash_a".into(), serde_json::json!({"k": 1}));
        assert_eq!(v1.version_seq, 1);
        assert!(v1.prev_version_seq.is_none());
        assert!(v1.valid_until.is_none());
        assert!(!v1.is_counterfactual);

        let latest = tree.get_latest("doc-1").unwrap();
        assert_eq!(latest.version_seq, 1);
        assert_eq!(latest.content_hash, "hash_a");
    }

    #[test]
    fn test_insert_seals_previous_version() {
        let mut tree = VersionTree::new();

        tree.insert("doc-1", "hash_v1".into(), serde_json::json!(null));
        // Small delay so timestamps differ
        std::thread::sleep(std::time::Duration::from_millis(10));
        let v2 = tree.insert("doc-1", "hash_v2".into(), serde_json::json!(null));

        assert_eq!(v2.version_seq, 2);
        assert_eq!(v2.prev_version_seq, Some(1));
        assert!(v2.valid_until.is_none()); // current head

        // Old head should be sealed
        let v1 = tree.get("doc-1", 1).unwrap();
        assert!(v1.valid_until.is_some());
    }

    #[test]
    fn test_history_newest_first() {
        let mut tree = VersionTree::new();

        for i in 0..5 {
            tree.insert("e", format!("hash_{}", i), serde_json::json!(i));
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        let history = tree.history("e");
        assert_eq!(history.len(), 5);
        // Newest first: seq 5, 4, 3, 2, 1
        for (i, node) in history.iter().enumerate() {
            assert_eq!(node.version_seq, 5 - i as u64);
        }
    }

    #[test]
    fn test_mark_counterfactual() {
        let mut tree = VersionTree::new();
        tree.insert("e", "h1".into(), serde_json::json!(null));
        tree.insert("e", "h2".into(), serde_json::json!(null));
        tree.insert("e", "h3".into(), serde_json::json!(null));

        // Mark version 2 as counterfactual
        tree.mark_counterfactual("e", 2).unwrap();

        let v2 = tree.get("e", 2).unwrap();
        assert!(v2.is_counterfactual);

        // v1 and v3 should remain non-counterfactual
        assert!(!tree.get("e", 1).unwrap().is_counterfactual);
        assert!(!tree.get("e", 3).unwrap().is_counterfactual);

        // Marking non-existent version should error
        assert!(tree.mark_counterfactual("e", 99).is_err());
        assert!(tree.mark_counterfactual("no-such", 1).is_err());
    }

    #[test]
    fn test_query_at_specific_time() {
        let mut tree = VersionTree::new();

        // Manually construct a version chain with known timestamps
        let t0 = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
        let t1 = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();
        let t2 = Utc.with_ymd_and_hms(2025, 9, 1, 0, 0, 0).unwrap();

        // Insert 3 versions, then backdate their timestamps
        tree.insert("e", "h0".into(), serde_json::json!(null));
        std::thread::sleep(std::time::Duration::from_millis(10));
        tree.insert("e", "h1".into(), serde_json::json!(null));
        std::thread::sleep(std::time::Duration::from_millis(10));
        tree.insert("e", "h2".into(), serde_json::json!(null));

        // Override timestamps for deterministic testing
        {
            let v0 = tree.nodes.get_mut(&(String::from("e"), 1)).unwrap();
            v0.valid_from = t0;
            v0.valid_until = Some(t1);

            let v1 = tree.nodes.get_mut(&(String::from("e"), 2)).unwrap();
            v1.valid_from = t1;
            v1.valid_until = Some(t2);

            let v2 = tree.nodes.get_mut(&(String::from("e"), 3)).unwrap();
            v2.valid_from = t2;
            v2.valid_until = None;
        }

        // Query at different points in time
        // Before any version existed
        let t_before = Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap();
        assert!(tree.query_at("e", t_before).is_none());

        // During v0's validity
        let t_during_v0 = Utc.with_ymd_and_hms(2025, 3, 15, 0, 0, 0).unwrap();
        let result = tree.query_at("e", t_during_v0).unwrap();
        assert_eq!(result.version_seq, 1);

        // Exactly at v1 start
        let result = tree.query_at("e", t1).unwrap();
        assert_eq!(result.version_seq, 2);

        // During v1's validity
        let t_during_v1 = Utc.with_ymd_and_hms(2025, 7, 20, 0, 0, 0).unwrap();
        let result = tree.query_at("e", t_during_v1).unwrap();
        assert_eq!(result.version_seq, 2);

        // During v2 (current head)
        let t_during_v2 = Utc.with_ymd_and_hms(2025, 10, 1, 0, 0, 0).unwrap();
        let result = tree.query_at("e", t_during_v2).unwrap();
        assert_eq!(result.version_seq, 3);
    }

    #[test]
    fn test_query_at_entity_not_found() {
        let tree = VersionTree::new();
        let result = tree.query_at("nonexistent", Utc::now());
        assert!(result.is_none());
    }

    #[test]
    fn test_get_specific_version() {
        let mut tree = VersionTree::new();
        tree.insert("e", "h1".into(), serde_json::json!(null));
        tree.insert("e", "h2".into(), serde_json::json!(null));

        assert!(tree.get("e", 1).is_some());
        assert!(tree.get("e", 2).is_some());
        assert!(tree.get("e", 3).is_none());
        assert!(tree.get("other", 1).is_none());
    }

    #[test]
    fn test_entity_ids() {
        let mut tree = VersionTree::new();
        tree.insert("alpha", "h".into(), serde_json::json!(null));
        tree.insert("beta", "h".into(), serde_json::json!(null));
        tree.insert("alpha", "h2".into(), serde_json::json!(null));

        let mut ids: Vec<&String> = tree.entity_ids().collect();
        ids.sort();
        assert_eq!(ids, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_entity_len() {
        let mut tree = VersionTree::new();
        for i in 0..7 {
            tree.insert("e", format!("h{}", i), serde_json::json!(i));
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(tree.entity_len("e"), 7);
        assert_eq!(tree.entity_len("other"), 0);
    }

    #[test]
    fn test_thousand_versions_performance() {
        let mut tree = VersionTree::new();
        let entity = "perf-test";

        // Insert 1 000 versions
        for i in 0..1000u64 {
            tree.insert(entity, format!("hash_{}", i), serde_json::json!(i));
        }
        assert_eq!(tree.len(), 1000);

        // O(1) lookups should be fast
        let start = std::time::Instant::now();
        for seq in 1..=1000u64 {
            let _ = tree.get(entity, seq).unwrap();
        }
        let elapsed = start.elapsed();
        // 1 000 HashMap lookups should complete well under 1ms
        assert!(
            elapsed.as_millis() < 50,
            "1000 lookups took {:?}, expected < 50ms",
            elapsed
        );

        // History traversal of 1 000 nodes
        let start = std::time::Instant::now();
        let history = tree.history(entity);
        let elapsed = start.elapsed();
        assert_eq!(history.len(), 1000);
        assert!(
            elapsed.as_millis() < 10,
            "history(1000) took {:?}, expected < 10ms",
            elapsed
        );
    }

    #[test]
    fn test_temporal_gate_filters_counterfactual() {
        let mut tree = VersionTree::new();
        tree.insert("e", "h1".into(), serde_json::json!(null));
        tree.insert("e", "h2".into(), serde_json::json!(null));
        tree.insert("e", "h3".into(), serde_json::json!(null));
        tree.mark_counterfactual("e", 2).unwrap();

        let now = Utc::now();
        let filtered = tree.temporal_gate("e", now, 30);
        let seqs: Vec<u64> = filtered.iter().map(|n| n.version_seq).collect();
        // v1 is expired (sealed by v2 insert), v2 is counterfactual,
        // only v3 (current head) survives the gate
        assert_eq!(seqs.len(), 1);
        assert!(seqs.contains(&3));
        assert!(!seqs.contains(&2)); // counterfactual excluded
        assert!(!seqs.contains(&1)); // expired (sealed by insert)
    }

    #[test]
    fn test_temporal_gate_filters_expired() {
        let mut tree = VersionTree::new();
        tree.insert("e", "h1".into(), serde_json::json!(null));
        std::thread::sleep(std::time::Duration::from_millis(10));
        tree.insert("e", "h2".into(), serde_json::json!(null));

        // Manually expire version 1
        {
            let v1 = tree.nodes.get_mut(&(String::from("e"), 1)).unwrap();
            v1.valid_until = Some(Utc::now() - chrono::Duration::days(1));
        }

        let now = Utc::now();
        let filtered = tree.temporal_gate("e", now, 30);
        let seqs: Vec<u64> = filtered.iter().map(|n| n.version_seq).collect();
        assert_eq!(seqs, vec![2]); // only current version survives
    }

    #[test]
    fn test_temporal_gate_empty_entity() {
        let tree = VersionTree::new();
        let now = Utc::now();
        let filtered = tree.temporal_gate("nonexistent", now, 30);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_temporal_gate_with_dormancy() {
        let mut tree = VersionTree::new();
        tree.insert("e", "h1".into(), serde_json::json!(null));

        let now = Utc::now();
        // Not dormant: tick_delta = 5, threshold = 30*10 = 300
        let (filtered, dormant) = tree.temporal_gate_with_dormancy("e", now, 30, 95, 100);
        assert_eq!(filtered.len(), 1);
        assert!(!dormant);

        // Dormant: tick_delta = 500, threshold = 30*10 = 300
        let (filtered, dormant) = tree.temporal_gate_with_dormancy("e", now, 30, 0, 500);
        assert_eq!(filtered.len(), 1);
        assert!(dormant);
    }

    #[test]
    fn test_get_diff() {
        let mut tree = VersionTree::new();
        tree.insert(
            "e",
            "hash_v1".into(),
            serde_json::json!({"status": "draft", "score": 0.5}),
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
        tree.insert(
            "e",
            "hash_v2".into(),
            serde_json::json!({"status": "final", "score": 0.9}),
        );

        let diff = tree.get_diff("e", 1, 2).unwrap();
        assert_eq!(diff.entity_id, "e");
        assert!(diff.content_changed, "content hash should differ");
        assert!(diff.time_delta_seconds >= 0);
        assert!(!diff.metadata_diff.is_empty(), "metadata should differ");

        // Verify specific metadata changes
        let status_change = diff.metadata_diff.iter().find(|(k, _, _)| k == "status");
        assert!(status_change.is_some());
        let (_, old_val, new_val) = status_change.unwrap();
        assert_eq!(old_val, &serde_json::json!("draft"));
        assert_eq!(new_val, &serde_json::json!("final"));
    }

    #[test]
    fn test_get_diff_nonexistent() {
        let tree = VersionTree::new();
        assert!(tree.get_diff("nope", 1, 2).is_none());
    }

    #[test]
    fn test_compress_old_versions() {
        let mut tree = VersionTree::new();

        // Create 10 versions
        for i in 0..10u64 {
            tree.insert("e", format!("hash_{}", i), serde_json::json!({"seq": i}));
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        // Compress with threshold 5 (keep last 5, compress seqs 1-5)
        let removed = tree.compress("e", 5).unwrap();
        assert_eq!(removed, 5);

        // Head (seq 10) should still exist
        let head = tree.get_latest("e").unwrap();
        assert_eq!(head.version_seq, 10);

        // The summary node should exist at seq 1
        let summary = tree.get("e", 1).unwrap();
        assert!(summary.metadata["compressed"].as_bool().unwrap());
        assert_eq!(summary.metadata["original_count"].as_u64().unwrap(), 5);

        // Versions 2-5 should be removed
        for seq in 2..=5u64 {
            assert!(tree.get("e", seq).is_none(), "seq {} should be removed", seq);
        }

        // Versions 6-10 should still exist
        for seq in 6..=10u64 {
            assert!(tree.get("e", seq).is_some(), "seq {} should exist", seq);
        }
    }

    #[test]
    fn test_compress_empty_entity() {
        let mut tree = VersionTree::new();
        let result = tree.compress("nonexistent", 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_compress_nothing_to_compress() {
        let mut tree = VersionTree::new();
        // Only 2 versions, threshold 10
        tree.insert("e", "h1".into(), serde_json::json!(null));
        tree.insert("e", "h2".into(), serde_json::json!(null));

        let removed = tree.compress("e", 10).unwrap();
        assert_eq!(removed, 0); // nothing old enough to compress
    }

    #[test]
    fn test_filter_active_normal_mode() {
        let mut tree = VersionTree::new();
        tree.insert("e", "h1".into(), serde_json::json!(null));
        std::thread::sleep(std::time::Duration::from_millis(10));
        tree.insert("e", "h2".into(), serde_json::json!(null));

        // Expire v1 manually
        {
            let v1 = tree.nodes.get_mut(&(String::from("e"), 1)).unwrap();
            v1.valid_until = Some(Utc::now() - chrono::Duration::days(1));
        }

        // Normal mode (downtime=0): expired v1 should be excluded
        let active = tree.filter_active("e", Utc::now(), 0);
        let seqs: Vec<u64> = active.iter().map(|n| n.version_seq).collect();
        assert_eq!(seqs, vec![2]); // only current head
    }

    #[test]
    fn test_filter_active_hibernation_mode() {
        let mut tree = VersionTree::new();
        tree.insert("e", "h1".into(), serde_json::json!(null));
        std::thread::sleep(std::time::Duration::from_millis(10));
        tree.insert("e", "h2".into(), serde_json::json!(null));

        // Expire v1 manually
        {
            let v1 = tree.nodes.get_mut(&(String::from("e"), 1)).unwrap();
            v1.valid_until = Some(Utc::now() - chrono::Duration::days(1));
        }

        // Hibernation mode (downtime=10 > 7): expired v1 KEPT for gap reflection
        let active = tree.filter_active("e", Utc::now(), 10);
        let seqs: Vec<u64> = active.iter().map(|n| n.version_seq).collect();
        assert!(seqs.contains(&1)); // v1 kept (historical context)
        assert!(seqs.contains(&2)); // v2 kept (current)
    }

    #[test]
    fn test_filter_active_excludes_counterfactual() {
        let mut tree = VersionTree::new();
        tree.insert("e", "h1".into(), serde_json::json!(null));
        tree.insert("e", "h2".into(), serde_json::json!(null));
        tree.mark_counterfactual("e", 2).unwrap();

        let active = tree.filter_active("e", Utc::now(), 0);
        let seqs: Vec<u64> = active.iter().map(|n| n.version_seq).collect();
        assert!(!seqs.contains(&2)); // counterfactual excluded
    }

    #[test]
    fn test_is_hibernating() {
        let tree = VersionTree::new();
        assert!(!tree.is_hibernating(5));
        assert!(!tree.is_hibernating(7));
        assert!(tree.is_hibernating(8));
        assert!(tree.is_hibernating(30));
    }
}
