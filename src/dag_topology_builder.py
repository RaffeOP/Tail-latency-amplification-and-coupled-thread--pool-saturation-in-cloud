"""
src/dag_topology_builder.py
Constructs canonical bounded microservice DAG topologies:
1. T1: Serial Chains (D in {2,4,6,8}, F=1)
2. T2: Bounded Scatter-Gather Trees (D in {2,3,4}, F in {2,4,8}, |V| <= 32)
3. T3: Diamond In-Cast Shared-Leaf Hotspot DAGs
4. T4: DeathStarBench Sub-Graphs (Social Network & Hotel Reservation)
Computes critical-path floors for DAC-Shed deadline propagation.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
Target: Journal of Cloud Computing (Scopus Q1) / Cluster Computing
"""

from typing import Dict, List, Set, Tuple
import networkx as nx

def build_serial_chain(depth: int) -> nx.DiGraph:
    """Builds a linear serial chain: v0 -> v1 -> ... -> v_{depth-1}."""
    G = nx.DiGraph()
    for i in range(depth):
        G.add_node(i, service_name=f"svc_chain_d{i}")
        if i > 0:
            G.add_edge(i - 1, i)
    return G

def build_bounded_scatter_gather(depth: int, fanout: int, max_vertices: int = 32) -> nx.DiGraph:
    """
    Constructs a scatter-gather tree with bounded total vertices (|V| <= max_vertices)
    to prevent combinatorial F^D explosion.
    """
    G = nx.DiGraph()
    G.add_node(0, service_name="gateway_v0")
    current_tier = [0]
    next_node_id = 1

    for d in range(1, depth):
        next_tier = []
        for parent in current_tier:
            for f in range(fanout):
                if next_node_id >= max_vertices:
                    break
                child_id = next_node_id
                G.add_node(child_id, service_name=f"svc_tree_d{d}_id{child_id}")
                G.add_edge(parent, child_id)
                next_tier.append(child_id)
                next_node_id += 1
            if next_node_id >= max_vertices:
                break
        if not next_tier:
            break
        current_tier = next_tier

    return G

def build_diamond_incast_dag(depth: int, width: int, shared_leaves: int = 2) -> nx.DiGraph:
    """
    Creates W parallel middle-tier pipelines converging onto 1-2 shared leaf services.
    Models hotspot contention where leaf queueing backpressures all upstream branches.
    """
    G = nx.DiGraph()
    G.add_node(0, service_name="ingress_gateway")
    curr_id = 1

    # Ingress branches out to width W
    middle_tier_starts = []
    middle_tier_ends = []

    for w in range(width):
        prev_node = 0
        for d in range(1, depth - 1):
            node_id = curr_id
            curr_id += 1
            G.add_node(node_id, service_name=f"pipe_w{w}_d{d}")
            G.add_edge(prev_node, node_id)
            prev_node = node_id
            if d == 1:
                middle_tier_starts.append(node_id)
        middle_tier_ends.append(prev_node)

    # All branches converge to shared leaves
    leaf_nodes = []
    for l in range(shared_leaves):
        leaf_id = curr_id
        curr_id += 1
        G.add_node(leaf_id, service_name=f"shared_leaf_db_{l}")
        leaf_nodes.append(leaf_id)
        for end_node in middle_tier_ends:
            G.add_edge(end_node, leaf_id)

    return G

def build_deathstar_social_network() -> nx.DiGraph:
    """
    Mirrors DeathStarBench Social Network ComposePost / ReadHomeTimeline call graph.
    12 Microservices with mixed fan-out + caching & storage tiers.
    """
    G = nx.DiGraph()
    services = [
        "nginx_gateway", "compose_post", "text_service", "user_timeline",
        "post_storage", "user_service", "media_service", "unique_id",
        "redis_timeline", "redis_post", "memcached_text", "mongodb_post"
    ]
    for i, name in enumerate(services):
        G.add_node(i, service_name=name)

    # Compose post flow
    G.add_edge(0, 1)  # nginx -> compose_post
    G.add_edge(1, 2)  # compose_post -> text_service
    G.add_edge(1, 6)  # compose_post -> media_service
    G.add_edge(1, 7)  # compose_post -> unique_id
    G.add_edge(1, 3)  # compose_post -> user_timeline
    G.add_edge(1, 4)  # compose_post -> post_storage
    G.add_edge(2, 10) # text_service -> memcached_text
    G.add_edge(3, 8)  # user_timeline -> redis_timeline
    G.add_edge(4, 9)  # post_storage -> redis_post
    G.add_edge(4, 11) # post_storage -> mongodb_post
    G.add_edge(1, 5)  # compose_post -> user_service

    return G

def compute_critical_path_floor(G: nx.DiGraph, baseline_proc_times: Dict[int, float] = None) -> Dict[int, float]:
    """
    Precomputes T_min_rem(v) for every vertex via reverse topological sort.
    Used by DAC-Shed for cross-hop SLA deadline budget verification.
    """
    if baseline_proc_times is None:
        baseline_proc_times = {n: 1.0 for n in G.nodes()}

    t_rem: Dict[int, float] = {}
    top_order = list(nx.topological_sort(G))

    for node in reversed(top_order):
        children = list(G.successors(node))
        if not children:
            t_rem[node] = baseline_proc_times.get(node, 1.0)
        else:
            max_child = max(t_rem[c] for c in children)
            t_rem[node] = baseline_proc_times.get(node, 1.0) + max_child

    return t_rem

def build_deathstar_hotel_reservation() -> nx.DiGraph:
    """
    Synthetic 8-vertex sub-graph shaped after the hotel search and reservation paths.
    Vertices 5 (rate) and 6 (user) are shared leaves, so their visit counts exceed one.
    """
    G = nx.DiGraph()
    services = ["frontend", "search", "profile", "reservation", "geo", "rate", "user", "recommendation"]
    for i, name in enumerate(services):
        G.add_node(i, service_name=name)
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 7), (1, 4), (1, 5), (2, 6), (3, 5), (3, 6)])
    return G

def build_topology(name: str, args: dict) -> nx.DiGraph:
    """Dispatches a topology family name and its arguments to the matching builder."""
    if name == "chain":
        return build_serial_chain(args["depth"])
    if name == "scatter_gather":
        return build_bounded_scatter_gather(args["depth"], args["fanout"])
    if name == "diamond":
        return build_diamond_incast_dag(args["depth"], args["width"])
    if name == "deathstar_social":
        return build_deathstar_social_network()
    if name == "deathstar_hotel":
        return build_deathstar_hotel_reservation()
    raise ValueError(f"unknown topology family: {name}")
