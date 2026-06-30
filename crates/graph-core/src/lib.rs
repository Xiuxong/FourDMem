//! graph-core: L1 atomic fact graph with hot-cognition weights
//!
//! - Node attributes: utility_score, activation_weight, last_active_tick
//! - Edge attributes: conflict_weight
//! - Concurrent snapshot reads via parking_lot RwLock
//! - Dual-format persistence: bincode (cache) + JSON (human-readable/Git)

pub mod graph;
pub mod tda;
pub mod version_tree;

// Re-export L1 public API at crate root.
pub use graph::{EdgeAttr, GraphError, L1Graph, NodeAttr, ShelfLife};
pub use tda::{PhaseTransitionSignal, TdaAnalyzer, TopologyMetrics};
pub use petgraph::stable_graph::NodeIndex;

// Re-export Version Tree public API at crate root.
pub use version_tree::{VersionDiff, VersionError, VersionNode, VersionTree};
pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T, E = Error> = std::result::Result<T, E>;
