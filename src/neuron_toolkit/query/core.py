from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np

from neuron_toolkit._types import TensorMap
from neuron_toolkit._utils import ShapeInfo, _GraphShim
from neuron_toolkit.backends.base import BaseParser
from neuron_toolkit.rewriter import NeuronRewriter

if TYPE_CHECKING:
    from neuron_toolkit.pattern import MatchResult, Pattern

log = logging.getLogger("neuron_toolkit.query")

MAX_REPR_NODES = 15


def _get_node_attrs(
    node: object, backend: BaseParser | None = None
) -> dict[str, object]:
    """Backend-agnostic attribute extraction."""
    if backend and hasattr(backend, "get_node_attrs"):
        return backend.get_node_attrs(node)

    # Fallback for TFLite (TFLiteNode usually has .attrs)
    attrs = getattr(node, "attrs", None)
    if isinstance(attrs, dict) and not hasattr(node, "attribute"):
        return attrs

    # Fallback for ONNX (NodeProto has .attribute)
    if hasattr(node, "attribute"):
        from neuron_toolkit.backends.onnx.utils import _node_attrs  # noqa: PLC0415

        return _node_attrs(cast(Any, node))

    return {}


class NeuronQuery:
    """Fluent API for querying and filtering model nodes.

    NeuronQuery allows you to traverse a model graph using a set of chainable
    filters and transformations.

    Example:
        >>> graph = NeuronGraph.load("model.proto")
        >>> q = graph.query().op("Conv").attr("group", 1)
        >>> results = q.findall(my_pattern)
    """

    def __init__(
        self,
        nodes: Sequence[object],
        tensor_map: TensorMap,
        all_nodes: Sequence[object],
        graph_inputs: set[str] | None = None,
        graph_outputs: set[str] | None = None,
        shape_info: ShapeInfo | None = None,
        backend: object | None = None,
    ) -> None:
        """Initialize a NeuronQuery."""
        self.nodes = list(nodes)
        self.tensor_map = tensor_map
        self.all_nodes = list(all_nodes)
        self.graph_inputs = graph_inputs or set()
        self.graph_outputs = graph_outputs or set()
        self.shape_info = shape_info or {}
        self._backend = backend

    @cached_property
    def output_map(self) -> dict[str, object]:
        """Return a mapping from output tensor names to their producing nodes."""
        return {out: n for n in self.all_nodes for out in getattr(n, "output", [])}

    @cached_property
    def _nx_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for n in self.all_nodes:
            name = getattr(n, "name", "")
            g.add_node(name, proto=n)
            node_inputs = getattr(n, "input", [])
            for inp in node_inputs:
                if parent := self.output_map.get(inp):
                    parent_name = getattr(parent, "name", "")
                    g.add_edge(parent_name, name, tensor=inp)
        return g

    @cached_property
    def _node_to_idx(self) -> dict[str, int]:
        """Global topological index of each node in the full graph."""
        try:
            order = list(nx.topological_sort(self._nx_graph))
            return {name: i for i, name in enumerate(order)}
        except nx.NetworkXCyclicError:
            # Fallback to original order if cycles exist
            return {
                getattr(n, "name", f"node_{i}"): i for i, n in enumerate(self.all_nodes)
            }

    def _clone(self, nodes: Sequence[object]) -> NeuronQuery:
        new = self.__class__(
            nodes,
            self.tensor_map,
            self.all_nodes,
            self.graph_inputs,
            self.graph_outputs,
            self.shape_info,
            backend=self._backend,
        )
        # Share expensive graph caches
        for attr in ("output_map", "_nx_graph", "_node_to_idx"):
            if attr in self.__dict__:
                new.__dict__[attr] = self.__dict__[attr]
        return new

    def _params(self, node: object) -> TensorMap:
        node_inputs = getattr(node, "input", [])
        return {i: self.tensor_map[i] for i in node_inputs if i in self.tensor_map}

    # --- Filters ---

    def filter(self, predicate: Callable[[object], bool]) -> NeuronQuery:
        """Filter nodes using a custom predicate."""
        return self._clone([n for n in self.nodes if predicate(n)])

    def op(self, op_type: str) -> NeuronQuery:
        """Alias for find_by_op_type."""
        return self.find_by_op_type(op_type)

    def name(self, name: str, *, exact: bool = False) -> NeuronQuery:
        """Alias for find_by_name."""
        return self.find_by_name(name, exact=exact)

    def rank(self, rank: int) -> NeuronQuery:
        """Filter nodes by their output tensor rank."""

        def _match(n: object) -> bool:
            outputs = getattr(n, "output", [])
            if not outputs:
                return False
            # Check shape_info (which now includes graph inputs)
            r, _ = self.shape_info.get(outputs[0], (None, None))
            return r == rank

        return self.filter(_match)

    def dtype(self, dtype: str) -> NeuronQuery:
        """Filter nodes by their output tensor data type."""

        def _match(n: object) -> bool:
            outputs = getattr(n, "output", [])
            if not outputs:
                return False
            _, d = self.shape_info.get(outputs[0], (None, None))
            return d == dtype

        return self.filter(_match)

    def attr(self, name: str, value: object = None) -> NeuronQuery:
        """Alias for find_by_attribute."""
        return self.find_by_attribute(name, value)

    def find_by_op_type(self, op_type: str) -> NeuronQuery:
        """Filter nodes by their op_type."""
        return self.filter(lambda n: getattr(n, "op_type", None) == op_type)

    def find_by_name(self, name: str, *, exact: bool = False) -> NeuronQuery:
        """Filter nodes by name (substring match by default)."""
        low = name.lower()
        return self.filter(
            lambda n: (
                getattr(n, "name", "") == name
                if exact
                else low in getattr(n, "name", "").lower()
            )
        )

    def find_by_tensor(self, name: str) -> NeuronQuery:
        """Filter nodes that consume or produce a specific tensor."""
        return self.filter(
            lambda n: (
                name in getattr(n, "input", []) or name in getattr(n, "output", [])
            )
        )

    def find_by_param_name(self, name: str, *, exact: bool = False) -> NeuronQuery:
        """Filter nodes that have a weight tensor with a specific name."""
        low = name.lower()
        return self.filter(
            lambda n: any(
                p == name if exact else low in p.lower() for p in self._params(n)
            )
        )

    def find_by_attribute(self, name: str, value: object = None) -> NeuronQuery:
        """Find nodes by attribute name and optionally value."""

        def _match(n: object) -> bool:
            attrs = _get_node_attrs(n, cast(Any, self._backend))
            if name not in attrs:
                return False
            if value is None:
                return True
            v = attrs[name]
            if callable(value):
                return cast(Callable[[Any], bool], value)(v)
            if isinstance(v, np.ndarray) or isinstance(value, np.ndarray):
                return np.array_equal(cast(Any, v), cast(Any, value))
            return v == value

        return self.filter(_match)

    # --- Traversal ---

    def _traverse(self, method: str, max_depth: int = 100) -> NeuronQuery:
        g = self._nx_graph
        if not self.nodes:
            return self._clone([])

        if method == "successors":
            targets = {
                s for n in self.nodes for s in g.successors(getattr(n, "name", ""))
            }
        elif method == "predecessors":
            targets = {
                p for n in self.nodes for p in g.predecessors(getattr(n, "name", ""))
            }
        else:
            # Optimized multi-source traversal
            rev = g.reverse(copy=False) if method == "ancestors" else g
            sources = [getattr(n, "name", "") for n in self.nodes]
            # Use multi-source BFS
            visited = set()
            queue = [(s, 0) for s in sources]
            while queue:
                u, d = queue.pop(0)
                if u not in visited:
                    visited.add(u)
                    if d < max_depth:
                        queue.extend(
                            (v, d + 1) for v in rev.successors(u) if v not in visited
                        )
            targets = visited - set(sources)

        return self._clone([g.nodes[name]["proto"] for name in targets])

    def children(self) -> NeuronQuery:
        """Return direct children of selected nodes."""
        return self._traverse("successors")

    def outputs(self) -> NeuronQuery:
        """Alias for children()."""
        return self.children()

    def parents(self) -> NeuronQuery:
        """Return direct parents of selected nodes."""
        return self._traverse("predecessors")

    def inputs(self) -> NeuronQuery:
        """Alias for parents()."""
        return self.parents()

    def ancestors(self, max_depth: int = 100) -> NeuronQuery:
        """Return all ancestors within *max_depth*."""
        return self._traverse("ancestors", max_depth)

    def descendants(self, max_depth: int = 100) -> NeuronQuery:
        """Return all descendants within *max_depth*."""
        return self._traverse("descendants", max_depth)

    # --- Entry / Exit ---

    @property
    def entry_nodes(self) -> NeuronQuery:
        """Return nodes that consume graph inputs."""
        return self._clone(
            [
                n
                for n in self.all_nodes
                if any(i in self.graph_inputs for i in getattr(n, "input", []))
            ]
        )

    @property
    def output_nodes(self) -> NeuronQuery:
        """Return nodes that produce graph outputs."""
        return self._clone(
            [
                n
                for n in self.all_nodes
                if any(o in self.graph_outputs for o in getattr(n, "output", []))
            ]
        )

    # --- Param access ---

    def has_params(self) -> NeuronQuery:
        """Filter nodes that have weight tensors."""
        return self.filter(lambda n: bool(self._params(n)))

    def tensor(self, name: str | None = None) -> object:
        """Retrieve parameter tensors for the selected nodes."""
        if name is not None:
            return self.tensor_map.get(name)
        if len(self.nodes) == 1:
            return self._params(self.nodes[0])
        return {
            getattr(n, "name", f"node_{i}"): p
            for i, n in enumerate(self.nodes)
            if (p := self._params(n))
        }

    def matches(self, pattern: Pattern) -> NeuronQuery:
        """Return nodes that are the start of a match for *pattern*."""
        from neuron_toolkit.pattern import PatternDetector  # noqa: PLC0415

        shim = _GraphShim(self.all_nodes, self.tensor_map, self.shape_info)
        matched = []
        for node in self.nodes:
            det = PatternDetector(shim, start_node=node)
            det.output_to_node = self.output_map
            if det.match(pattern) is not None:
                matched.append(node)
        return self._clone(matched)

    def follow(self, pattern: Pattern) -> NeuronQuery:
        """Follow a pattern forward from the current nodes.

        Returns the root nodes of matches that incorporate the current nodes.
        """
        from neuron_toolkit.pattern import PatternDetector  # noqa: PLC0415

        shim = _GraphShim(self.all_nodes, self.tensor_map, self.shape_info)
        results = []
        current_ids = {id(n) for n in self.nodes}

        # We search for all matches in the graph
        det = PatternDetector(shim)
        det.output_to_node = self.output_map
        all_matches = det.find_all(pattern)

        results = [
            m.start for m in all_matches if any(id(n) in current_ids for n in m.nodes)
        ]

        return self._clone(results)

    def match_results(self, pattern: Pattern) -> list[MatchResult]:
        """Return all MatchResult objects for matches starting at these nodes."""
        from neuron_toolkit.pattern import PatternDetector  # noqa: PLC0415

        shim = _GraphShim(self.all_nodes, self.tensor_map, self.shape_info)
        results = []
        for node in self.nodes:
            det = PatternDetector(shim, start_node=node)
            det.output_to_node = self.output_map
            if r := det.match(pattern):
                r.tensor_map = self.tensor_map
                results.append(r)
        return results

    def select(self, pattern: Pattern) -> list[MatchResult]:
        """Alias for match_results — find all pattern matches."""
        return self.match_results(pattern)

    def where(self, pattern: Pattern) -> NeuronQuery:
        """Filter nodes that are the start of a match for *pattern*."""
        return self.matches(pattern)

    def replace(
        self,
        pattern: Pattern,
        new_op: str,
        name: str | None = None,
        **attrs: object,
    ) -> NeuronRewriter:
        """Replace matches of *pattern* with a new operator."""
        if not self._backend:
            msg = "Query has no backend; cannot perform rewrite."
            raise RuntimeError(msg)

        # Get the underlying backend rewriter
        impl = cast(Any, self._backend).rewriter()
        rewriter = NeuronRewriter(impl)

        for r in self.select(pattern):
            cast(Any, rewriter).replace_from_result(r, new_op, name=name, **attrs)
        return rewriter

    def to_pattern(self) -> Pattern:
        """Convert the first node of this query into a reusable Pattern."""
        from neuron_toolkit.pattern import Pattern  # noqa: PLC0415

        if not self.nodes:
            msg = "Cannot convert empty query to pattern"
            raise ValueError(msg)

        node = self.nodes[0]
        # Basic conversion: op_type + attributes
        p = Pattern.op(getattr(node, "op_type", ""))
        attrs = _get_node_attrs(node)
        if attrs:
            p = p.where(**attrs)
        return p

    # --- Ordering ---

    def topological_sort(self) -> NeuronQuery:
        """Return nodes in global topological order."""
        idx_map = self._node_to_idx
        sorted_nodes = sorted(
            self.nodes,
            key=lambda n: idx_map.get(getattr(n, "name", ""), 999999),
        )
        return self._clone(sorted_nodes)

    def is_topologically_sorted(self) -> bool:
        """Return True if current node list is in topological order."""
        idx_map = self._node_to_idx
        last_idx = -1
        for n in self.nodes:
            curr_idx = idx_map.get(getattr(n, "name", ""), 999999)
            if curr_idx < last_idx:
                return False
            last_idx = curr_idx
        return True

    def apply(self, fn: Callable[[object, TensorMap], object]) -> NeuronQuery:
        """Apply *fn* to each node and its parameters."""
        for node in self.nodes:
            fn(node, self._params(node))
        return self

    # --- Set Operations ---

    def union(self, other: NeuronQuery) -> NeuronQuery:
        """Return union of nodes with another query."""
        # Use object identity for set operations to handle unnamed nodes
        seen_ids = {id(n) for n in self.nodes}
        return self._clone(
            self.nodes + [n for n in other.nodes if id(n) not in seen_ids]
        )

    def intersection(self, other: NeuronQuery) -> NeuronQuery:
        """Return intersection of nodes with another query."""
        other_ids = {id(n) for n in other.nodes}
        return self._clone([n for n in self.nodes if id(n) in other_ids])

    def difference(self, other: NeuronQuery) -> NeuronQuery:
        """Return nodes in this query but not in the other."""
        other_ids = {id(n) for n in other.nodes}
        return self._clone([n for n in self.nodes if id(n) not in other_ids])

    def __or__(self, other: NeuronQuery) -> NeuronQuery:
        return self.union(other)

    def __and__(self, other: NeuronQuery) -> NeuronQuery:
        return self.intersection(other)

    def __sub__(self, other: NeuronQuery) -> NeuronQuery:
        return self.difference(other)

    # --- Accessors ---

    @property
    def single_node(self) -> object:
        """Return the single node in this query."""
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

    def first(self) -> object | None:
        """Return the first node or None."""
        return self.nodes[0] if self.nodes else None

    def last(self) -> object | None:
        """Return the last node or None."""
        return self.nodes[-1] if self.nodes else None

    def count(self) -> int:
        """Return the number of nodes."""
        return len(self.nodes)

    def is_empty(self) -> bool:
        """Return True if empty."""
        return not self.nodes

    def op_types(self) -> list[str]:
        """Return a list of unique op types in this query."""
        return list(dict.fromkeys(getattr(n, "op_type", "") for n in self.nodes))

    def __iter__(self) -> Iterator[object]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __bool__(self) -> bool:
        return bool(self.nodes)

    def __getitem__(self, i: int | slice) -> NeuronQuery:
        if isinstance(i, slice):
            return self._clone(self.nodes[i])
        return self._clone([self.nodes[i]])

    def __repr__(self) -> str:
        if not self.nodes:
            return "NeuronQuery(empty)"
        lines = [f"NeuronQuery: {len(self.nodes)} nodes"]
        for i, n in enumerate(self.nodes[:MAX_REPR_NODES]):
            params = self._params(n)
            p = ", ".join(f"{k}:{list(v.shape)}" for k, v in params.items()) or "-"
            name = getattr(n, "name", "<unnamed>")
            op_type = getattr(n, "op_type", "<unknown>")
            lines.append(f"  [{i:3}] {op_type:16} {name} (params: {p})")
        if len(self.nodes) > MAX_REPR_NODES:
            lines.append(f"  ... (+{len(self.nodes) - MAX_REPR_NODES} more)")
        return "\n".join(lines)
