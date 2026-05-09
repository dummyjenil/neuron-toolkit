from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
from onnx.onnx_pb import NodeProto

from onnx_toolkit._types import TensorMap
from onnx_toolkit._utils import ShapeInfo, _GraphShim, _node_attrs
from onnx_toolkit.rewriter import GraphRewriter

if TYPE_CHECKING:
    from onnx_toolkit.pattern import MatchResult, Pattern

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

    @cached_property
    def _node_to_idx(self) -> dict[str, int]:
        """Global topological index of each node in the full graph."""
        try:
            order = list(nx.topological_sort(self._nx_graph))
            return {name: i for i, name in enumerate(order)}
        except nx.NetworkXCyclicError:
            # Fallback to original order if cycles exist
            return {n.name: i for i, n in enumerate(self.all_nodes)}

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
        for attr in ("_output_map", "_nx_graph", "_node_to_idx"):
            if attr in self.__dict__:
                new.__dict__[attr] = self.__dict__[attr]
        return new

    def _params(self, node: NodeProto) -> TensorMap:
        return {i: self.tensor_map[i] for i in node.input if i in self.tensor_map}

    # --- Filters ---

    def filter(self, predicate: Callable[[NodeProto], bool]) -> ONNXQuery:
        """Filter nodes using a custom predicate."""
        return self._clone([n for n in self.nodes if predicate(n)])

    def op(self, op_type: str) -> ONNXQuery:
        """Alias for find_by_op_type."""
        return self.find_by_op_type(op_type)

    def name(self, name: str, *, exact: bool = False) -> ONNXQuery:
        """Alias for find_by_name."""
        return self.find_by_name(name, exact=exact)

    def rank(self, rank: int) -> ONNXQuery:
        """Filter nodes by their output tensor rank."""

        def _match(n: NodeProto) -> bool:
            if not n.output:
                return False
            # Check shape_info (which now includes graph inputs)
            r, _ = self.shape_info.get(n.output[0], (None, None))
            return r == rank

        return self.filter(_match)

    def dtype(self, dtype: str) -> ONNXQuery:
        """Filter nodes by their output tensor data type."""

        def _match(n: NodeProto) -> bool:
            if not n.output:
                return False
            _, d = self.shape_info.get(n.output[0], (None, None))
            return d == dtype

        return self.filter(_match)

    def attr(self, name: str, value: Any = None) -> ONNXQuery:
        """Alias for find_by_attribute."""
        return self.find_by_attribute(name, value)

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
            if callable(value):
                return value(v)
            if isinstance(v, np.ndarray) or isinstance(value, np.ndarray):
                return np.array_equal(v, value)
            return v == value

        return self.filter(_match)

    # --- Traversal ---

    def _traverse(self, method: str, max_depth: int = 100) -> ONNXQuery:
        g = self._nx_graph
        if not self.nodes:
            return self._clone([])

        if method == "successors":
            targets = {s for n in self.nodes for s in g.successors(n.name)}
        elif method == "predecessors":
            targets = {p for n in self.nodes for p in g.predecessors(n.name)}
        else:
            # Optimized multi-source traversal
            rev = g.reverse(copy=False) if method == "ancestors" else g
            sources = [n.name for n in self.nodes]
            # Use multi-source BFS
            visited = set()
            queue = [(s, 0) for s in sources]
            while queue:
                u, d = queue.pop(0)
                if u not in visited:
                    visited.add(u)
                    if d < max_depth:
                        for v in rev.successors(u):
                            if v not in visited:
                                queue.append((v, d + 1))
            targets = visited - set(sources)

        return self._clone([g.nodes[name]["proto"] for name in targets])

    def children(self) -> ONNXQuery:
        """Return direct children of selected nodes."""
        return self._traverse("successors")

    def outputs(self) -> ONNXQuery:
        """Alias for children()."""
        return self.children()

    def parents(self) -> ONNXQuery:
        """Return direct parents of selected nodes."""
        return self._traverse("predecessors")

    def inputs(self) -> ONNXQuery:
        """Alias for parents()."""
        return self.parents()

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
        return {
            n.name or f"node_{i}": p for i, n in enumerate(self.nodes) if (p := self._params(n))
        }

    def matches(self, pattern: Pattern) -> ONNXQuery:
        """Return nodes that are the start of a match for *pattern*."""
        from onnx_toolkit.pattern import PatternDetector

        shim = _GraphShim(self.all_nodes, self.tensor_map, self.shape_info)
        matched = []
        for node in self.nodes:
            det = PatternDetector(shim, start_node=node)
            det._output_to_node = self._output_map
            if det.match(pattern) is not None:
                matched.append(node)
        return self._clone(matched)

    def follow(self, pattern: Pattern) -> ONNXQuery:
        """Follow a pattern forward from the current nodes.
        Returns the root nodes of matches that incorporate the current nodes.
        """
        from onnx_toolkit.pattern import PatternDetector

        shim = _GraphShim(self.all_nodes, self.tensor_map, self.shape_info)
        results = []
        current_ids = {id(n) for n in self.nodes}

        # We search for all matches in the graph
        det = PatternDetector(shim)
        det._output_to_node = self._output_map
        all_matches = det.find_all(pattern)

        for m in all_matches:
            # If any of our current nodes are part of this match, we include the match root (start)
            if any(id(n) in current_ids for n in m.nodes):
                results.append(m.start)

        return self._clone(results)

    def match_results(self, pattern: Pattern) -> list[MatchResult]:
        """Return all MatchResult objects for matches starting at these nodes."""
        from onnx_toolkit.pattern import PatternDetector

        shim = _GraphShim(self.all_nodes, self.tensor_map, self.shape_info)
        results = []
        for node in self.nodes:
            det = PatternDetector(shim, start_node=node)
            det._output_to_node = self._output_map
            if r := det.match(pattern):
                r.tensor_map = self.tensor_map
                results.append(r)
        return results

    def select(self, pattern: Pattern) -> list[MatchResult]:
        """Alias for match_results — find all pattern matches starting from these nodes."""
        return self.match_results(pattern)

    def where(self, pattern: Pattern) -> ONNXQuery:
        """Filter nodes that are the start of a match for *pattern*."""
        return self.matches(pattern)

    def replace(
        self,
        pattern: Pattern,
        new_op: str,
        name: str | None = None,
        **attrs: Any,
    ) -> GraphRewriter:
        """Replace matches of *pattern* starting at these nodes with a new operator."""
        # We need a dummy parser to initialize the rewriter
        from onnx_toolkit.parser import ONNXParser

        # This is a bit hacky but keeps things decoupled
        class _ProxyParser:
            def __init__(self, nodes, tensor_map):
                from onnx import helper

                self.model = helper.make_model(helper.make_graph(nodes, "temp", [], []))
                self.nodes = nodes
                self.tensor_map = tensor_map

        # In reality, we should probably have GraphRewriter accept nodes/tensor_map directly
        # But for now, let's try to get the parent parser if possible or create a proxy
        # Since we don't store the parser, we'll create a minimal proxy
        proxy = _ProxyParser(self.all_nodes, self.tensor_map)
        rewriter = GraphRewriter(proxy)  # type: ignore

        for r in self.select(pattern):
            rewriter.replace_from_result(r, new_op, name=name, **attrs)
        return rewriter

    def to_pattern(self) -> Pattern:
        """Convert the first node of this query into a reusable Pattern.

        If multiple nodes are present, only the first one is used.
        """
        from onnx_toolkit.pattern import Pattern

        if not self.nodes:
            msg = "Cannot convert empty query to pattern"
            raise ValueError(msg)

        node = self.nodes[0]
        # Basic conversion: op_type + attributes
        p = Pattern.op(node.op_type)
        attrs = _node_attrs(node)
        if attrs:
            p = p.where(**attrs)
        return p

    # --- Ordering ---

    def topological_sort(self) -> ONNXQuery:
        """Return nodes in global topological order."""
        idx_map = self._node_to_idx
        sorted_nodes = sorted(self.nodes, key=lambda n: idx_map.get(n.name, 999999))
        return self._clone(sorted_nodes)

    def is_topologically_sorted(self) -> bool:
        """Return True if current node list is in topological order."""
        idx_map = self._node_to_idx
        last_idx = -1
        for n in self.nodes:
            curr_idx = idx_map.get(n.name, 999999)
            if curr_idx < last_idx:
                return False
            last_idx = curr_idx
        return True

    def apply(self, fn: Callable[[NodeProto, TensorMap], Any]) -> ONNXQuery:
        """Apply *fn* to each node and its parameters."""
        for node in self.nodes:
            fn(node, self._params(node))
        return self

    # --- Set Operations ---

    def union(self, other: ONNXQuery) -> ONNXQuery:
        """Return union of nodes with another query."""
        # Use object identity for set operations to handle unnamed nodes
        seen_ids = {id(n) for n in self.nodes}
        return self._clone(self.nodes + [n for n in other.nodes if id(n) not in seen_ids])

    def intersection(self, other: ONNXQuery) -> ONNXQuery:
        """Return intersection of nodes with another query."""
        other_ids = {id(n) for n in other.nodes}
        return self._clone([n for n in self.nodes if id(n) in other_ids])

    def difference(self, other: ONNXQuery) -> ONNXQuery:
        """Return nodes in this query but not in the other."""
        other_ids = {id(n) for n in other.nodes}
        return self._clone([n for n in self.nodes if id(n) not in other_ids])

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
        if isinstance(i, slice):
            return self._clone(self.nodes[i])
        return self._clone([self.nodes[i]])

    def __repr__(self) -> str:
        if not self.nodes:
            return "ONNXQuery(empty)"
        lines = [f"ONNXQuery: {len(self.nodes)} nodes"]
        for i, n in enumerate(self.nodes[:MAX_REPR_NODES]):
            params = self._params(n)
            p = ", ".join(f"{k}:{list(v.shape)}" for k, v in params.items()) or "-"
            lines.append(f"  [{i:3}] {n.op_type:16} {n.name or '<unnamed>'} (params: {p})")
        if len(self.nodes) > MAX_REPR_NODES:
            lines.append(f"  ... (+{len(self.nodes) - MAX_REPR_NODES} more)")
        return "\n".join(lines)
