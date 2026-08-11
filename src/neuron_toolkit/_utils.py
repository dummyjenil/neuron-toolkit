"""neuron_toolkit._utils — internal helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

log = logging.getLogger("neuron_toolkit")


# ShapeInfo: value_name → (rank | None, dtype_str | None)
ShapeInfo = dict[str, tuple[int | None, str | None]]


# ---------------------------------------------------------------------------
# Graph shim
# ---------------------------------------------------------------------------


class _GraphShim:
    """Lightweight stand-in for ModelProto used inside PatternDetector."""

    def __init__(
        self,
        nodes: Sequence[object],
        tensor_map: Mapping[str, object],
        shape_info: ShapeInfo | None = None,
        backend: object | None = None,
    ) -> None:
        self.nodes = nodes
        self.tensor_map = tensor_map
        self.shape_info: ShapeInfo = shape_info or {}
        self.backend = backend


def trace_subgraph_boundaries(
    nodes: Sequence[object],
    tensor_map: Mapping[str, object],
    start_points: list[str | object] | str | object,
    end_points: list[str | object] | str | object,
    original_outputs: set[str] | None = None,
) -> tuple[list[object], list[str], list[str]]:
    """Trace and extract the boundary inputs, outputs and nodes of a subgraph.

    Using networkx directed graph connectivity, we find all operators that lie on any path
    from start_points to end_points.
    """
    import networkx as nx

    # Normalize start/end points
    if not isinstance(start_points, list):
        start_points = [start_points]
    if not isinstance(end_points, list):
        end_points = [end_points]

    g = nx.DiGraph()
    node_name_to_idx = {}

    for idx, node in enumerate(nodes):
        node_name = getattr(node, "name", None)
        if node_name:
            node_name_to_idx[node_name] = idx

        g.add_node(idx, type="op")
        for inp in getattr(node, "input", []):
            if inp:
                g.add_node(inp, type="tensor")
                g.add_edge(inp, idx)
        for out in getattr(node, "output", []):
            if out:
                g.add_node(out, type="tensor")
                g.add_edge(idx, out)

    node_to_idx = {id(n): i for i, n in enumerate(nodes)}

    def find_node_idx(n: object) -> int | None:
        return node_to_idx.get(id(n))

    def resolve_point(pt: object) -> object:
        if isinstance(pt, str):
            if pt in g:
                return pt
            if pt in node_name_to_idx:
                return node_name_to_idx[pt]
        else:
            val = find_node_idx(pt)
            if val is not None:
                return val
        return pt

    S = {resolve_point(p) for p in start_points}
    E = {resolve_point(p) for p in end_points}

    S = {s for s in S if s in g}
    E = {e for e in E if e in g}

    if not S or not E:
        msg = "Could not resolve any valid start or end points in the graph."
        raise ValueError(msg)

    # Find descendants of S
    descendants = set()
    for s in S:
        descendants.add(s)
        descendants.update(nx.descendants(g, s))

    # Find ancestors of E
    ancestors = set()
    for e in E:
        ancestors.add(e)
        ancestors.update(nx.ancestors(g, e))

    subgraph_nodes = descendants.intersection(ancestors)
    kept_op_ids = [n for n in subgraph_nodes if g.nodes[n].get("type") == "op"]
    kept_ops = [nodes[op_id] for op_id in kept_op_ids]

    # Ensure all inputs and outputs of kept operators are in subgraph_nodes
    for op in kept_ops:
        for inp in getattr(op, "input", []):
            if inp:
                subgraph_nodes.add(inp)
        for out in getattr(op, "output", []):
            if out:
                subgraph_nodes.add(out)

    subgraph_tensors = {n for n in subgraph_nodes if g.nodes[n].get("type") == "tensor"}

    # Find producers for each tensor
    producers = {}
    for op in kept_ops:
        op_idx = find_node_idx(op)
        for out in getattr(op, "output", []):
            if out:
                producers[out] = op_idx

    # Find consumers for each tensor
    original_consumers = {}
    for idx, node in enumerate(nodes):
        for inp in getattr(node, "input", []):
            if inp:
                original_consumers.setdefault(inp, []).append(idx)

    # Subgraph inputs are tensors consumed but not produced within the subgraph (and not in tensor_map)
    new_inputs = []
    for t in subgraph_tensors:
        if t not in producers and t not in tensor_map:
            new_inputs.append(t)

    # Subgraph outputs
    new_outputs = []
    for t in subgraph_tensors:
        if t in E:
            new_outputs.append(t)
            continue
        if t in producers:
            if producers[t] in E:
                new_outputs.append(t)
                continue
            consumers = original_consumers.get(t, [])
            has_external_consumers = any(c not in kept_op_ids for c in consumers)
            if has_external_consumers or (original_outputs and t in original_outputs):
                new_outputs.append(t)

    new_inputs = sorted(set(new_inputs))
    new_outputs = sorted(set(new_outputs))

    return kept_ops, new_inputs, new_outputs
