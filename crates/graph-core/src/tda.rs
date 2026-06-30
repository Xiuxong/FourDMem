//! Topological Data Analysis (TDA) for L1 graph
//!
//! Computes Betti numbers to detect when the knowledge graph reaches
//! a "critical complexity" threshold that may trigger cross-domain insights.

use serde::{Deserialize, Serialize};

use crate::graph::L1Graph;

/// Topological metrics computed from the L1 graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyMetrics {
    pub node_count: usize,
    pub edge_count: usize,
    pub betti_0: usize,
    pub betti_1: usize,
    pub density: f64,
    pub avg_degree: f64,
    pub isolated_ratio: f64,
    pub clustering_coeff: f64,
}

impl Default for TopologyMetrics {
    fn default() -> Self {
        Self {
            node_count: 0,
            edge_count: 0,
            betti_0: 0,
            betti_1: 0,
            density: 0.0,
            avg_degree: 0.0,
            isolated_ratio: 0.0,
            clustering_coeff: 0.0,
        }
    }
}

/// Result of a phase-transition analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransitionSignal {
    pub triggered: bool,
    pub metrics: TopologyMetrics,
    pub signals: Vec<String>,
}

/// TDA analyzer for the L1 graph.
pub struct TdaAnalyzer {
    pub betti_1_threshold: u32,
    pub density_threshold: f64,
    pub isolated_ratio_threshold: f64,
}

impl Default for TdaAnalyzer {
    fn default() -> Self {
        Self {
            betti_1_threshold: 45,
            density_threshold: 0.15,
            isolated_ratio_threshold: 0.3,
        }
    }
}

impl TdaAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_thresholds(
        betti_1_threshold: u32,
        density_threshold: f64,
        isolated_ratio_threshold: f64,
    ) -> Self {
        Self {
            betti_1_threshold,
            density_threshold,
            isolated_ratio_threshold,
        }
    }

    /// Compute topological metrics for the L1 graph.
    pub fn compute_metrics(&self, graph: &L1Graph) -> TopologyMetrics {
        let node_count = graph.node_count();
        let edge_count = graph.edge_count();

        if node_count == 0 {
            return TopologyMetrics::default();
        }

        let betti_0 = self.count_connected_components(graph);
        let betti_1 = if edge_count + betti_0 > node_count {
            edge_count + betti_0 - node_count
        } else {
            0
        };

        let max_edges = (node_count * (node_count - 1)) as f64 / 2.0;
        let density = if max_edges > 0.0 {
            edge_count as f64 / max_edges
        } else {
            0.0
        };

        let avg_degree = if node_count > 0 {
            2.0 * edge_count as f64 / node_count as f64
        } else {
            0.0
        };

        let isolated = self.count_isolated(graph);
        let isolated_ratio = isolated as f64 / node_count as f64;
        let clustering_coeff = self.estimate_clustering_coeff(graph);

        TopologyMetrics {
            node_count,
            edge_count,
            betti_0,
            betti_1,
            density,
            avg_degree,
            isolated_ratio,
            clustering_coeff,
        }
    }

    /// Check if the graph is approaching a phase transition.
    pub fn check_phase_transition(&self, graph: &L1Graph) -> PhaseTransitionSignal {
        let metrics = self.compute_metrics(graph);
        let mut signals = Vec::new();

        if metrics.betti_1 >= self.betti_1_threshold as usize {
            signals.push(format!(
                "betti_1_critical: {} >= {}",
                metrics.betti_1, self.betti_1_threshold
            ));
        }

        if metrics.density > self.density_threshold {
            signals.push(format!(
                "high_density: {:.4} > {}",
                metrics.density, self.density_threshold
            ));
        }

        if metrics.isolated_ratio > self.isolated_ratio_threshold {
            signals.push(format!(
                "high_isolation: {:.3} > {}",
                metrics.isolated_ratio, self.isolated_ratio_threshold
            ));
        }

        PhaseTransitionSignal {
            triggered: !signals.is_empty(),
            metrics,
            signals,
        }
    }

    fn count_connected_components(&self, graph: &L1Graph) -> usize {
        let node_indices = graph.node_indices();
        let mut visited = std::collections::HashSet::new();
        let mut components = 0;

        for start in &node_indices {
            if visited.contains(start) {
                continue;
            }
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(*start);
            visited.insert(*start);

            while let Some(node) = queue.pop_front() {
                for neighbor in graph.get_neighbors(node) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
            components += 1;
        }
        components
    }

    fn count_isolated(&self, graph: &L1Graph) -> usize {
        graph
            .node_indices()
            .iter()
            .filter(|&&idx| graph.get_neighbors(idx).is_empty())
            .count()
    }

    fn estimate_clustering_coeff(&self, graph: &L1Graph) -> f64 {
        let node_indices = graph.node_indices();
        if node_indices.is_empty() {
            return 0.0;
        }

        let sample_size = node_indices.len().min(100);
        let step = if node_indices.len() > sample_size {
            node_indices.len() / sample_size
        } else {
            1
        };

        let mut total_triangles = 0usize;
        let mut total_possible = 0usize;

        for (i, node) in node_indices.iter().enumerate() {
            if i % step != 0 {
                continue;
            }

            let neighbors = graph.get_neighbors(*node);
            if neighbors.len() < 2 {
                continue;
            }

            let neighbor_set: std::collections::HashSet<_> = neighbors.iter().collect();
            let mut triangles = 0;

            for neighbor in &neighbors {
                for neighbor2 in graph.get_neighbors(*neighbor) {
                    if neighbor2 != *node && neighbor_set.contains(&neighbor2) {
                        triangles += 1;
                    }
                }
            }
            triangles /= 2;

            let possible = neighbors.len() * (neighbors.len() - 1) / 2;
            total_triangles += triangles;
            total_possible += possible;
        }

        if total_possible > 0 {
            total_triangles as f64 / total_possible as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EdgeAttr, L1Graph, NodeAttr};

    #[test]
    fn test_empty_graph() {
        let analyzer = TdaAnalyzer::new();
        let graph = L1Graph::new();
        let metrics = analyzer.compute_metrics(&graph);
        assert_eq!(metrics.node_count, 0);
        assert_eq!(metrics.betti_0, 0);
        assert_eq!(metrics.betti_1, 0);
    }

    #[test]
    fn test_single_node() {
        let analyzer = TdaAnalyzer::new();
        let mut graph = L1Graph::new();
        graph.add_node(NodeAttr::new("test"));
        let metrics = analyzer.compute_metrics(&graph);
        assert_eq!(metrics.node_count, 1);
        assert_eq!(metrics.betti_0, 1);
        assert_eq!(metrics.betti_1, 0);
        assert_eq!(metrics.isolated_ratio, 1.0);
    }

    #[test]
    fn test_linear_chain() {
        let analyzer = TdaAnalyzer::new();
        let mut graph = L1Graph::new();
        let a = graph.add_node(NodeAttr::new("A"));
        let b = graph.add_node(NodeAttr::new("B"));
        let c = graph.add_node(NodeAttr::new("C"));
        graph.add_edge(a, b, EdgeAttr::supports()).unwrap();
        graph.add_edge(b, c, EdgeAttr::supports()).unwrap();

        let metrics = analyzer.compute_metrics(&graph);
        assert_eq!(metrics.node_count, 3);
        assert_eq!(metrics.betti_0, 1);
        assert_eq!(metrics.betti_1, 0);
    }

    #[test]
    fn test_triangle_cycle() {
        let analyzer = TdaAnalyzer::new();
        let mut graph = L1Graph::new();
        let a = graph.add_node(NodeAttr::new("A"));
        let b = graph.add_node(NodeAttr::new("B"));
        let c = graph.add_node(NodeAttr::new("C"));
        graph.add_edge(a, b, EdgeAttr::supports()).unwrap();
        graph.add_edge(b, c, EdgeAttr::supports()).unwrap();
        graph.add_edge(c, a, EdgeAttr::supports()).unwrap();

        let metrics = analyzer.compute_metrics(&graph);
        assert_eq!(metrics.node_count, 3);
        assert_eq!(metrics.betti_0, 1);
        assert_eq!(metrics.betti_1, 1);
    }

    #[test]
    fn test_two_components() {
        let analyzer = TdaAnalyzer::new();
        let mut graph = L1Graph::new();
        let a = graph.add_node(NodeAttr::new("A"));
        let b = graph.add_node(NodeAttr::new("B"));
        graph.add_edge(a, b, EdgeAttr::supports()).unwrap();
        graph.add_node(NodeAttr::new("C"));

        let metrics = analyzer.compute_metrics(&graph);
        assert_eq!(metrics.node_count, 3);
        assert_eq!(metrics.betti_0, 2);
    }

    #[test]
    fn test_phase_transition_detection() {
        let analyzer = TdaAnalyzer::with_thresholds(3, 0.5, 0.5);
        let mut graph = L1Graph::new();

        let nodes: Vec<_> = (0..5)
            .map(|i| graph.add_node(NodeAttr::new(format!("node_{}", i))))
            .collect();

        for i in 0..5 {
            for j in (i + 1)..5 {
                graph.add_edge(nodes[i], nodes[j], EdgeAttr::supports()).unwrap();
            }
        }

        let signal = analyzer.check_phase_transition(&graph);
        assert!(signal.triggered);
        assert!(!signal.signals.is_empty());
    }
}
