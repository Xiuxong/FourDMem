//! Cross-layer reference integrity checker (T-2.5)
//!
//! Scans L1 nodes' `source_l0_refs` and verifies each referenced L0
//! evidence row exists. Orphan references are flagged in the report.

use serde::{Deserialize, Serialize};

use crate::l0::L0Store;
use graph_core::L1Graph;

// ── Report types ───────────────────────────────────────────────────────────────

/// A single orphan reference: an L1 node points to a non-existent L0 row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrphanRef {
    /// The L1 node label that contains the dangling reference.
    pub node_label: String,
    /// The L0 evidence ID that does not exist.
    pub missing_l0_id: i64,
}

/// Result of an integrity check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityReport {
    /// Total number of L1 nodes scanned.
    pub nodes_scanned: usize,
    /// Total number of `source_l0_refs` checked.
    pub refs_checked: usize,
    /// Orphan references found.
    pub orphans: Vec<OrphanRef>,
}

impl IntegrityReport {
    /// Whether the graph has any broken cross-layer references.
    pub fn is_clean(&self) -> bool {
        self.orphans.is_empty()
    }
}

// ── Checker ────────────────────────────────────────────────────────────────────

/// Run an integrity check: verify every L1 node's `source_l0_refs` points
/// to an existing L0 evidence row.
///
/// This is a simple function, not a background daemon — call it explicitly
/// when you want to audit cross-layer consistency.
pub fn check_integrity(l0: &L0Store, graph: &L1Graph) -> IntegrityReport {
    let mut orphans = Vec::new();
    let mut refs_checked = 0usize;
    let mut nodes_scanned = 0usize;

    for ni in graph.node_indices() {
        let node = match graph.get_node(ni) {
            Some(n) => n,
            None => continue,
        };
        nodes_scanned += 1;
        for &l0_id in &node.source_l0_refs {
            refs_checked += 1;
            match l0.exists(l0_id) {
                Ok(true) => {} // reference valid
                Ok(false) => {
                    orphans.push(OrphanRef {
                        node_label: node.label.clone(),
                        missing_l0_id: l0_id,
                    });
                }
                Err(_) => {
                    // DB error — treat as orphan for safety
                    orphans.push(OrphanRef {
                        node_label: node.label.clone(),
                        missing_l0_id: l0_id,
                    });
                }
            }
        }
    }

    IntegrityReport {
        nodes_scanned,
        refs_checked,
        orphans,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use graph_core::graph::NodeAttr;

    #[test]
    fn test_clean_graph() {
        let l0 = L0Store::open_memory().unwrap();
        // Insert two L0 rows
        let id1 = l0
            .append("test", "test-model", "s1", "user", "evidence 1", &serde_json::Value::Null)
            .unwrap();
        let id2 = l0
            .append("test", "test-model", "s1", "user", "evidence 2", &serde_json::Value::Null)
            .unwrap();

        let mut graph = L1Graph::new();
        graph.add_node(NodeAttr::with_sources("fact A", vec![id1]));
        graph.add_node(NodeAttr::with_sources("fact B", vec![id2]));

        let report = check_integrity(&l0, &graph);
        assert!(report.is_clean());
        assert_eq!(report.nodes_scanned, 2);
        assert_eq!(report.refs_checked, 2);
        assert!(report.orphans.is_empty());
    }

    #[test]
    fn test_orphan_detected() {
        let l0 = L0Store::open_memory().unwrap();
        let id1 = l0
            .append("test", "test-model", "s1", "user", "evidence 1", &serde_json::Value::Null)
            .unwrap();

        let mut graph = L1Graph::new();
        // id1 exists, id999 does not
        graph.add_node(NodeAttr::with_sources("fact A", vec![id1, 999]));

        let report = check_integrity(&l0, &graph);
        assert!(!report.is_clean());
        assert_eq!(report.orphans.len(), 1);
        assert_eq!(report.orphans[0].missing_l0_id, 999);
        assert_eq!(report.orphans[0].node_label, "fact A");
    }

    #[test]
    fn test_no_refs() {
        let l0 = L0Store::open_memory().unwrap();
        let mut graph = L1Graph::new();
        // Node with no source_l0_refs
        graph.add_node(NodeAttr::new("orphan fact"));

        let report = check_integrity(&l0, &graph);
        assert!(report.is_clean());
        assert_eq!(report.refs_checked, 0);
    }

    #[test]
    fn test_empty_graph() {
        let l0 = L0Store::open_memory().unwrap();
        let graph = L1Graph::new();

        let report = check_integrity(&l0, &graph);
        assert!(report.is_clean());
        assert_eq!(report.nodes_scanned, 0);
    }

    #[test]
    fn test_multiple_orphans() {
        let l0 = L0Store::open_memory().unwrap();
        // No L0 rows at all

        let mut graph = L1Graph::new();
        graph.add_node(NodeAttr::with_sources("fact A", vec![100, 200]));
        graph.add_node(NodeAttr::with_sources("fact B", vec![300]));

        let report = check_integrity(&l0, &graph);
        assert!(!report.is_clean());
        assert_eq!(report.orphans.len(), 3);
        assert_eq!(report.refs_checked, 3);
    }
}
