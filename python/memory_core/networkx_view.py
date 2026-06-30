"""NetworkX view converter for FourDMem L1 graph.

Converts the internal L1 graph (accessed via PyO3 bindings) into a
NetworkX DiGraph for analysis, visualization, and graph algorithms.

Usage:
    from memory_core.networkx_view import to_networkx
    G = to_networkx(engine)
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
"""

import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx


def to_networkx(engine: Any) -> "nx.DiGraph":
    """Convert the L1 graph to a NetworkX DiGraph.

    Each node carries attributes: label, utility_score, last_active_tick,
    shelf_life, embedding_dim.

    Each edge carries attributes: relation_type, conflict_weight.

    Args:
        engine: A FourDMemEngine instance (from py-bindings).

    Returns:
        A NetworkX DiGraph with the L1 graph structure.
    """
    import networkx as nx

    G = nx.DiGraph()

    # Get all node indices
    indices = engine.graph_node_indices()

    for idx in indices:
        node_json = engine.graph_get_node(idx)
        node_data = json.loads(node_json)
        G.add_node(idx, **node_data)

    # Get all edges
    edges = engine.graph_get_edges()
    for src, dst, rel_type, conflict_weight in edges:
        G.add_edge(src, dst, relation_type=rel_type, conflict_weight=conflict_weight)

    return G


def get_conflict_subgraph(G: "nx.DiGraph") -> "nx.DiGraph":
    """Extract only the conflict edges from the graph.

    Useful for analyzing which facts contradict each other.

    Args:
        G: A NetworkX DiGraph (as returned by to_networkx).

    Returns:
        A subgraph containing only edges with conflict_weight > 0.
    """
    import networkx as nx

    conflict_edges = [
        (u, v, d)
        for u, v, d in G.edges(data=True)
        if d.get("conflict_weight", 0) > 0
    ]

    H = nx.DiGraph()
    for u, v, d in conflict_edges:
        # Copy node attributes from original graph
        H.add_node(u, **G.nodes[u])
        H.add_node(v, **G.nodes[v])
        H.add_edge(u, v, **d)

    return H


def get_high_utility_nodes(G: "nx.DiGraph", threshold: float = 0.7) -> list:
    """Return nodes with utility_score above the threshold.

    Args:
        G: A NetworkX DiGraph.
        threshold: Minimum utility score to include.

    Returns:
        List of (node_id, attributes) tuples.
    """
    return [
        (n, d)
        for n, d in G.nodes(data=True)
        if d.get("utility_score", 0) >= threshold
    ]
