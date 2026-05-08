from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from functools import cached_property
from typing import Any, Callable

import networkx as nx
import numpy as np
from onnx.onnx_pb import NodeProto

from onnx_toolkit._types import TensorMap
from onnx_toolkit._utils import ShapeInfo, _GraphShim, _node_attrs
from onnx_toolkit.pattern import MatchResult, Pattern, PatternDetector

log = logging.getLogger("onnx_toolkit.query")

MAX_REPR_NODES = 15


class ONNXQuery:
    """Lazy, chainable view over a subset of ONNX graph nodes."""

    def __init__(
        self,
        nodes: Sequence[NodeProto],
        tensor_map: TensorMap,
        all_nodes: Sequence[NodeProto],
        graph_inputs: set[str] | None = None,
        graph_outputs: set[str] | None = None,
        shape_info: ShapeInfo | None = None,
    ) -> None:
        """Initialize ONNXQuery with nodes and graph metadata."""
        self.nodes = list(nodes)
        self.tensor_map = tensor_map
        self.all_nodes = list(all_nodes)
        self.graph_inputs = graph_inputs or set()
        self.graph_outputs = graph_outputs or set()
        self.shape_info = shape_info or {}

    @cached_property
    def _output_map(self) -> dict[str, NodeProto]:
        return {out: n for n in self.all_nodes for out in n.output}

    @cached_property
    def _nx_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for n in self.all_nodes:
            g.add_node(n.name, proto=n)
            for inp in n.input:
                if parent := self._output_map.get(inp):
                    g.add_edge(parent.name, n.name, tensor=inp)
        return g

    def _clone(self, nodes: Sequence[NodeProto]) -> ONNXQuery:
        new = self.__class__(
            nodes,
            self.tensor_map,
            self.all_nodes,
            self.graph_inputs,
            self.graph_outputs,
            self.shape_info,
        )
        # Share expensive graph caches
        for attr in ("_output_map", "_nx_graph"):
            if attr in self.__dict__:
                new.__dict__[attr] = self.__dict__[attr]
        return new

    def _params(self, node: NodeProto) -> TensorMap:
        return {i: self.tensor_map[i] for i in node.input if i in self.tensor_map}

    # --- Filters ---

    def filter(self, predicate: Callable[[NodeProto], bool]) -> ONNXQuery:
        """Filter nodes using a custom predicate."""
        return self._clone([n for n in self.nodes if predicate(n)])

    def find_by_op_type(self, op_type: str) -> ONNXQuery:
        """Filter nodes by their op_type."""
        return self.filter(lambda n: n.op_type == op_type)

    def find_by_name(self, name: str, *, exact: bool = False) -> ONNXQuery:
        """Filter nodes by name (substring match by default)."""
        low = name.lower()
        return self.filter(lambda n: n.name == name if exact else low in n.name.lower())

    def find_by_tensor(self, name: str) -> ONNXQuery:
        """Filter nodes that consume or produce a specific tensor."""
        return self.filter(lambda n: name in n.input or name in n.output)

    def find_by_param_name(self, name: str, *, exact: bool = False) -> ONNXQuery:
        """Filter nodes that have a weight tensor with a specific name."""
        low = name.lower()
        return self.filter(
            lambda n: any(p == name if exact else low in p.lower() for p in self._params(n))
        )

    def find_by_attribute(self, name: str, value: Any = None) -> ONNXQuery:
        """Find nodes by attribute name and optionally value."""

        def _match(n: NodeProto) -> bool:
            attrs = _node_attrs(n)
            if name not in attrs:
                return False
            if value is None:
                return True
            v = attrs[name]
            return value(v) if callable(value) else v == value

        return self.filter(_match)

    # --- Traversal ---

    def _traverse(self, method: str, max_depth: int = 100) -> ONNXQuery:
        g, targets = self._nx_graph, set()
        for n in self.nodes:
            if method == "successors":
                targets.update(g.successors(n.name))
            elif method == "predecessors":
                targets.update(g.predecessors(n.name))
            else:
                rev = g.reverse(copy=False) if method == "ancestors" else g
                lengths = nx.single_source_shortest_path_length(rev, n.name, cutoff=max_depth)
                targets.update(lengths.keys())

        if method in ("ancestors", "descendants"):
            targets -= {n.name for n in self.nodes}

        return self._clone([g.nodes[name]["proto"] for name in targets])

    def children(self) -> ONNXQuery:
        """Return direct children of selected nodes."""
        return self._traverse("successors")

    def parents(self) -> ONNXQuery:
        """Return direct parents of selected nodes."""
        return self._traverse("predecessors")

    def ancestors(self, max_depth: int = 100) -> ONNXQuery:
        """Return all ancestors within *max_depth*."""
        return self._traverse("ancestors", max_depth)

    def descendants(self, max_depth: int = 100) -> ONNXQuery:
        """Return all descendants within *max_depth*."""
        return self._traverse("descendants", max_depth)

    # --- Entry / Exit ---

    @property
    def entry_nodes(self) -> ONNXQuery:
        """Return nodes that consume graph inputs."""
        return self._clone(
            [n for n in self.all_nodes if any(i in self.graph_inputs for i in n.input)]
        )

    @property
    def output_nodes(self) -> ONNXQuery:
        """Return nodes that produce graph outputs."""
        return self._clone(
            [n for n in self.all_nodes if any(o in self.graph_outputs for o in n.output)]
        )

    # --- Param access ---

    def has_params(self) -> ONNXQuery:
        """Filter nodes that have weight tensors."""
        return self.filter(lambda n: bool(self._params(n)))

    def tensor(self, name: str | None = None) -> object:
        """Retrieve parameter tensors for the selected nodes."""
        if name is not None:
            return self.tensor_map.get(name)
        if len(self.nodes) == 1:
            return self._params(self.nodes[0])
        return {n.name: p for n in self.nodes if (p := self._params(n))}

    def matches(self, pattern: Pattern) -> ONNXQuery:
        """Return nodes that are the start of a match for *pattern*."""
        shim = _GraphShim(self.all_nodes, self.tensor_map, self.shape_info)
        matched = []
        for node in self.nodes:
            det = PatternDetector(shim, start_node=node)
            det._output_to_node = self._output_map
            if det.match(pattern) is not None:
                matched.append(node)
        return self._clone(matched)

    def match_results(self, pattern: Pattern) -> list[MatchResult]:
        """Return all MatchResult objects for matches starting at these nodes."""
        shim = _GraphShim(self.all_nodes, self.tensor_map, self.shape_info)
        results = []
        for node in self.nodes:
            det = PatternDetector(shim, start_node=node)
            det._output_to_node = self._output_map
            if r := det.match(pattern):
                r.tensor_map = self.tensor_map
                results.append(r)
        return results

    # --- Ordering ---

    def topological_sort(self) -> ONNXQuery:
        """Return nodes in topological order."""
        subgraph = self._nx_graph.subgraph([n.name for n in self.nodes])
        try:
            order = list(nx.topological_sort(subgraph))
            return self._clone([self._nx_graph.nodes[name]["proto"] for name in order])
        except nx.NetworkXCyclicError:
            msg = "Cycle detected in selected subgraph."
            raise ValueError(msg) from None

    def is_topologically_sorted(self) -> bool:
        """Return True if current node list is in topological order."""
        try:
            self.topological_sort()
            name_to_idx = {n.name: i for i, n in enumerate(self.nodes)}
            sub = self._nx_graph.subgraph(name_to_idx.keys())
            return all(name_to_idx[u] <= name_to_idx[v] for u, v in sub.edges())
        except ValueError:
            return False

    def apply(self, fn: Callable[[NodeProto, TensorMap], Any]) -> ONNXQuery:
        """Apply *fn* to each node and its parameters."""
        for node in self.nodes:
            fn(node, self._params(node))
        return self

    # --- Set Operations ---

    def union(self, other: ONNXQuery) -> ONNXQuery:
        """Return union of nodes with another query."""
        seen = {n.name for n in self.nodes}
        return self._clone(self.nodes + [n for n in other.nodes if n.name not in seen])

    def intersection(self, other: ONNXQuery) -> ONNXQuery:
        """Return intersection of nodes with another query."""
        other_names = {n.name for n in other.nodes}
        return self._clone([n for n in self.nodes if n.name in other_names])

    def difference(self, other: ONNXQuery) -> ONNXQuery:
        """Return nodes in this query but not in the other."""
        other_names = {n.name for n in other.nodes}
        return self._clone([n for n in self.nodes if n.name not in other_names])

    def __or__(self, other: ONNXQuery) -> ONNXQuery:
        return self.union(other)

    def __and__(self, other: ONNXQuery) -> ONNXQuery:
        return self.intersection(other)

    def __sub__(self, other: ONNXQuery) -> ONNXQuery:
        return self.difference(other)

    # --- Accessors ---

    @property
    def single_node(self) -> NodeProto:
        """Return the single node in this query. Raises ValueError if count != 1."""
        if len(self.nodes) != 1:
            msg = f"Expected 1 node, got {len(self.nodes)}"
            raise ValueError(msg)
        return self.nodes[0]

    @property
    def single_tensor(self) -> np.ndarray:
        """Return the single weight tensor for the single node in this query."""
        p = self._params(self.single_node)
        if len(p) != 1:
            msg = f"Expected 1 tensor, got {list(p.keys())}"
            raise ValueError(msg)
        return next(iter(p.values()))

    def first(self) -> NodeProto | None:
        return self.nodes[0] if self.nodes else None

    def last(self) -> NodeProto | None:
        return self.nodes[-1] if self.nodes else None

    def count(self) -> int:
        return len(self.nodes)

    def is_empty(self) -> bool:
        return not self.nodes

    def op_types(self) -> list[str]:
        return list(dict.fromkeys(n.op_type for n in self.nodes))

    def __iter__(self) -> Iterator[NodeProto]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __bool__(self) -> bool:
        return bool(self.nodes)

    def __getitem__(self, i: int | slice) -> ONNXQuery:
        nodes = self.nodes[i] if isinstance(i, slice) else [self.nodes[i]]
        return self._clone(nodes)

    def __repr__(self) -> str:
        if not self.nodes:
            return "ONNXQuery(empty)"
        lines = [f"ONNXQuery: {len(self.nodes)} nodes"]
        for i, n in enumerate(self.nodes[:MAX_REPR_NODES]):
            p = ", ".join(f"{k}:{list(v.shape)}" for k, v in self._params(n).items()) or "-"
            lines.append(f"  [{i:3}] {n.op_type:16} {n.name} (params: {p})")
        if len(self.nodes) > MAX_REPR_NODES:
            lines.append(f"  ... (+{len(self.nodes) - MAX_REPR_NODES} more)")
        return "\n".join(lines)
