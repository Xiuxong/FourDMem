//! L1 atomic fact graph — undirected semantic knowledge graph.
//!
//! Each node represents a structured fact extracted from L0 evidence.
//! Edges are semantically undirected ("A relates to B" ≡ "B relates to A").
//! Contradiction edges are directional and stored with conflict_weight.
//!
//! Uses `petgraph::StableUnGraph` for bidirectional traversal without
//! edge duplication. Node/edge indices remain stable after removals.

use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableUnGraph};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("node index {0:?} not found")]
    NodeNotFound(NodeIndex),
    #[error("edge index {0:?} not found")]
    EdgeNotFound(EdgeIndex),
    #[error("self-loops are not allowed")]
    SelfLoop,
}

// ── Shelf-life category ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShelfLife {
    Physical(u32),
    Subjective(u64),
    Immune,
}

impl Default for ShelfLife {
    fn default() -> Self { ShelfLife::Subjective(90) }
}

// ── Node attributes ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAttr {
    pub layer: u8,
    pub source_l0_refs: Vec<i64>,
    pub utility_score: f64,
    pub last_active_tick: u64,
    pub shelf_life: ShelfLife,
    pub label: String,
    pub embedding: Vec<f32>,
    pub origin_workspace: Option<String>,
    pub origin_type: String,
}

impl NodeAttr {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            layer: 1, source_l0_refs: Vec::new(), utility_score: 0.0,
            last_active_tick: 0, shelf_life: ShelfLife::default(),
            label: label.into(), embedding: Vec::new(),
            origin_workspace: None, origin_type: "vault_native".to_string(),
        }
    }
    pub fn with_sources(label: impl Into<String>, l0_refs: Vec<i64>) -> Self {
        Self { source_l0_refs: l0_refs, ..Self::new(label) }
    }
    pub fn with_origin(label: impl Into<String>, origin_ws: String, origin_type: String) -> Self {
        Self { origin_workspace: Some(origin_ws), origin_type, ..Self::new(label) }
    }
}

// ── Edge attributes ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeAttr {
    pub relation: String,        // "supports" | "contradicts" | "elaborates" | "related_to"
    pub conflict_weight: f64,    // 0.0–1.0, only meaningful for "contradicts"
    pub semantic_score: f64,     // 0.0–1.0, embedding similarity or confidence
}

impl EdgeAttr {
    pub fn supports() -> Self {
        Self { relation: "supports".into(), conflict_weight: 0.0, semantic_score: 0.5 }
    }
    pub fn contradicts(weight: f64) -> Self {
        Self { relation: "contradicts".into(), conflict_weight: weight.clamp(0.0, 1.0), semantic_score: 0.0 }
    }
    pub fn elaborates() -> Self {
        Self { relation: "elaborates".into(), conflict_weight: 0.0, semantic_score: 0.7 }
    }
    pub fn related_to(score: f64) -> Self {
        Self { relation: "related_to".into(), conflict_weight: 0.0, semantic_score: score.clamp(0.0, 1.0) }
    }
}

// ── L1 Graph ──────────────────────────────────────────────────────────────────

/// Undirected semantic graph of L1 atomic facts.
///
/// Edges are semantically bidirectional — "A relates to B" implies "B relates to A".
/// This correctly models the knowledge graph where relatedness is symmetric.
/// Contradiction edges are the exception: stored directionally with conflict_weight.
pub struct L1Graph {
    graph: StableUnGraph<NodeAttr, EdgeAttr>,
}

impl L1Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self { graph: StableUnGraph::default() }
    }

    pub fn node_count(&self) -> usize { self.graph.node_count() }
    pub fn edge_count(&self) -> usize { self.graph.edge_count() }

    /// Add a fact node. Returns its stable index.
    pub fn add_node(&mut self, attr: NodeAttr) -> NodeIndex {
        self.graph.add_node(attr)
    }

    /// Add an undirected edge. For "contradicts" edges, direction is advisory.
    pub fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, attr: EdgeAttr) -> Result<EdgeIndex, GraphError> {
        if a == b { return Err(GraphError::SelfLoop); }
        if self.graph.node_weight(a).is_none() { return Err(GraphError::NodeNotFound(a)); }
        if self.graph.node_weight(b).is_none() { return Err(GraphError::NodeNotFound(b)); }
        Ok(self.graph.add_edge(a, b, attr))
    }

    pub fn get_node(&self, idx: NodeIndex) -> Option<&NodeAttr> {
        self.graph.node_weight(idx)
    }

    pub fn get_node_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeAttr> {
        self.graph.node_weight_mut(idx)
    }

    /// All neighbors (undirected — both directions).
    pub fn get_neighbors(&self, idx: NodeIndex) -> Vec<NodeIndex> {
        self.graph.neighbors(idx).collect()
    }

    /// Search nodes by token overlap with label.
    pub fn search_nodes(&self, query: &str) -> Vec<(NodeIndex, &NodeAttr)> {
        let query_lower = query.to_lowercase();
        let tokens: Vec<&str> = query_lower.split_whitespace().collect();
        self.graph.node_indices()
            .filter_map(|ni| {
                let attr = &self.graph[ni];
                if tokens.iter().any(|t| attr.label.to_lowercase().contains(t)) {
                    Some((ni, attr))
                } else { None }
            })
            .collect()
    }

    pub fn node_indices(&self) -> Vec<NodeIndex> {
        self.graph.node_indices().collect()
    }

    /// Increment `last_active_tick` for a node (subjective time warmup).
    pub fn bump_tick(&mut self, idx: NodeIndex) -> Result<(), GraphError> {
        let node = self.graph.node_weight_mut(idx).ok_or(GraphError::NodeNotFound(idx))?;
        node.last_active_tick += 1;
        Ok(())
    }

    /// Adjust utility score by delta, clamped to [-1.0, 1.0].
    pub fn adjust_utility(&mut self, idx: NodeIndex, delta: f64) -> Result<f64, GraphError> {
        let node = self.graph.node_weight_mut(idx).ok_or(GraphError::NodeNotFound(idx))?;
        node.utility_score = (node.utility_score + delta).clamp(-1.0, 1.0);
        Ok(node.utility_score)
    }

    /// Find all neighbors with conflict_weight > 0.5.
    pub fn find_conflicts(&self, idx: NodeIndex) -> Vec<NodeIndex> {
        let mut conflicts: Vec<NodeIndex> = self.graph.edges(idx)
            .filter(|e| e.weight().conflict_weight > 0.5)
            .map(|e| {
                let (a, b) = self.graph.edge_endpoints(e.id()).unwrap();
                if a == idx { b } else { a }
            })
            .collect();
        conflicts.sort(); conflicts.dedup();
        conflicts
    }

    /// Approximate Betti-1 (cycle count). b1 ≈ |E| - |V| + |CC|.
    pub fn betti_1_approx(&self) -> u64 {
        let v = self.graph.node_count() as u64;
        if v == 0 { return 0; }
        let e = self.graph.edge_count() as u64;
        let cc = self.connected_components() as u64;
        e + cc - v
    }

    /// Count weakly connected components via BFS.
    fn connected_components(&self) -> usize {
        use std::collections::{HashSet, VecDeque};
        let mut visited: HashSet<NodeIndex> = HashSet::new();
        let mut components = 0;
        for node in self.graph.node_indices() {
            if visited.contains(&node) { continue; }
            components += 1;
            let mut queue = VecDeque::new();
            queue.push_back(node);
            visited.insert(node);
            while let Some(current) = queue.pop_front() {
                for neighbor in self.graph.neighbors(current) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        components
    }

    // ── Serialization ─────────────────────────────────────────────────────────

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let s = SerializableGraph::from_graph(self);
        serde_json::to_string_pretty(&s)
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let s: SerializableGraph = serde_json::from_str(json)?;
        Ok(s.into_graph())
    }

    pub fn to_bincode(&self) -> Vec<u8> {
        let s = SerializableGraph::from_graph(self);
        bincode::serialize(&s).expect("bincode serialize should not fail")
    }

    pub fn from_bincode(bytes: &[u8]) -> Result<Self, bincode::Error> {
        let s: SerializableGraph = bincode::deserialize(bytes)?;
        Ok(s.into_graph())
    }
}

impl Default for L1Graph {
    fn default() -> Self { Self::new() }
}

// ── Serializable helper ───────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct SerializableGraph {
    nodes: Vec<(usize, NodeAttr)>,
    edges: Vec<(usize, usize, EdgeAttr)>,
}

impl SerializableGraph {
    fn from_graph(g: &L1Graph) -> Self {
        let nodes = g.graph.node_indices()
            .map(|ni| (ni.index(), g.graph[ni].clone()))
            .collect();
        let edges = g.graph.edge_indices()
            .map(|ei| {
                let (src, dst) = g.graph.edge_endpoints(ei).unwrap();
                (src.index(), dst.index(), g.graph[ei].clone())
            })
            .collect();
        Self { nodes, edges }
    }

    fn into_graph(self) -> L1Graph {
        let mut graph = L1Graph::new();
        let mut idx_map = std::collections::HashMap::new();
        for (old_idx, attr) in self.nodes {
            let ni = graph.add_node(attr);
            idx_map.insert(old_idx, ni);
        }
        for (src_old, dst_old, attr) in self.edges {
            if let (Some(&src), Some(&dst)) = (idx_map.get(&src_old), idx_map.get(&dst_old)) {
                let _ = graph.add_edge(src, dst, attr);
            }
        }
        graph
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_graph_empty() {
        let g = L1Graph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert_eq!(g.betti_1_approx(), 0);
    }

    #[test]
    fn test_add_node() {
        let mut g = L1Graph::new();
        let n = g.add_node(NodeAttr::new("test"));
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.get_node(n).unwrap().label, "test");
    }

    #[test]
    fn test_add_edge_undirected() {
        let mut g = L1Graph::new();
        let a = g.add_node(NodeAttr::new("A"));
        let b = g.add_node(NodeAttr::new("B"));
        g.add_edge(a, b, EdgeAttr::related_to(0.8)).unwrap();
        assert_eq!(g.edge_count(), 1);

        // Undirected: neighbors work both ways
        assert!(g.get_neighbors(a).contains(&b));
        assert!(g.get_neighbors(b).contains(&a));
    }

    #[test]
    fn test_self_loop_rejected() {
        let mut g = L1Graph::new();
        let a = g.add_node(NodeAttr::new("A"));
        assert!(g.add_edge(a, a, EdgeAttr::supports()).is_err());
    }

    #[test]
    fn test_connected_components() {
        let mut g = L1Graph::new();
        let a = g.add_node(NodeAttr::new("A"));
        let b = g.add_node(NodeAttr::new("B"));
        let c = g.add_node(NodeAttr::new("C"));
        g.add_edge(a, b, EdgeAttr::supports()).unwrap();
        // A-B connected, C isolated → 2 components
        assert_eq!(g.betti_1_approx(), 0); // tree: E=1, V=3, CC=2 → 1+2-3=0
    }

    #[test]
    fn test_triangle_betti() {
        let mut g = L1Graph::new();
        let a = g.add_node(NodeAttr::new("A"));
        let b = g.add_node(NodeAttr::new("B"));
        let c = g.add_node(NodeAttr::new("C"));
        g.add_edge(a, b, EdgeAttr::supports()).unwrap();
        g.add_edge(b, c, EdgeAttr::supports()).unwrap();
        g.add_edge(c, a, EdgeAttr::supports()).unwrap();
        // Triangle: E=3, V=3, CC=1 → b1 = 3+1-3 = 1
        assert_eq!(g.betti_1_approx(), 1);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut g = L1Graph::new();
        let a = g.add_node(NodeAttr::new("fact A"));
        let b = g.add_node(NodeAttr::new("fact B"));
        g.add_edge(a, b, EdgeAttr::elaborates()).unwrap();

        let json = g.to_json().unwrap();
        let g2 = L1Graph::from_json(&json).unwrap();
        assert_eq!(g2.node_count(), 2);
        assert_eq!(g2.edge_count(), 1);
    }

    #[test]
    fn test_find_conflicts() {
        let mut g = L1Graph::new();
        let a = g.add_node(NodeAttr::new("A"));
        let b = g.add_node(NodeAttr::new("B"));
        g.add_edge(a, b, EdgeAttr::contradicts(0.8)).unwrap();
        let conflicts = g.find_conflicts(a);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0], b);
    }
}
