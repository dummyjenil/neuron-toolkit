"""
onnx_toolkit.query
===================
ONNXQuery — chainable, lazy view over a subset of ONNX graph nodes.

New in this version
-------------------
.topological_sort()     re-order nodes by Kahn's algorithm
.is_topologically_sorted()  check ordering
.apply(fn)              call fn(node, params) on every selected node
.match_results(pat)     like .matches() but returns List[MatchResult]
.to_dot()               Graphviz DOT string
.to_dataframe()         pandas DataFrame of node metadata
.to_onnx(path)          save selected subgraph as a standalone .onnx file
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Union

import numpy as np
from onnx.onnx_pb import NodeProto

from ._types import TensorMap
from ._utils import _GraphShim, _attr_value, ShapeInfo
from .pattern import Pattern, PatternDetector, MatchResult

log = logging.getLogger("onnx_toolkit.query")


class ONNXQuery:
    """
    A lazy, chainable view over a subset of ONNX graph nodes.

    Parameters
    ----------
    nodes        : selected nodes
    tensor_map   : weight tensors (initializers) keyed by name
    all_nodes    : every node in the graph (for traversal)
    graph_inputs : names of graph-level inputs
    graph_outputs: names of graph-level outputs
    shape_info   : output_name → (rank, dtype) from shape inference
    """

    def __init__(
        self,
        nodes:         List[NodeProto],
        tensor_map:    TensorMap,
        all_nodes:     List[NodeProto],
        graph_inputs:  Optional[Set[str]] = None,
        graph_outputs: Optional[Set[str]] = None,
        shape_info:    Optional[ShapeInfo] = None,
    ) -> None:
        self.nodes         = nodes
        self.tensor_map    = tensor_map
        self.all_nodes     = all_nodes
        self.graph_inputs  = graph_inputs  or set()
        self.graph_outputs = graph_outputs or set()
        self.shape_info    = shape_info    or {}
        self._output_to_node: Optional[Dict[str, NodeProto]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_output_map(self) -> Dict[str, NodeProto]:
        if self._output_to_node is None:
            self._output_to_node = {
                out: n for n in self.all_nodes for out in n.output
            }
        return self._output_to_node

    def _clone(self, nodes: List[NodeProto]) -> "ONNXQuery":
        q = ONNXQuery(nodes, self.tensor_map, self.all_nodes,
                      self.graph_inputs, self.graph_outputs, self.shape_info)
        q._output_to_node = self._output_to_node
        return q

    def _params(self, node: NodeProto) -> TensorMap:
        return {inp: self.tensor_map[inp] for inp in node.input if inp in self.tensor_map}

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def find_by_op_type(self, op_type: str) -> "ONNXQuery":
        """Filter to nodes whose op_type == *op_type* (case-sensitive)."""
        return self._clone([n for n in self.nodes if n.op_type == op_type])

    def find_by_name(self, name: str, *, exact: bool = False) -> "ONNXQuery":
        """Filter by node name (case-insensitive substring or exact match)."""
        if exact:
            return self._clone([n for n in self.nodes if n.name == name])
        low = name.lower()
        return self._clone([n for n in self.nodes if low in n.name.lower()])

    def find_by_tensor(self, tensor_name: str) -> "ONNXQuery":
        """Filter to nodes that consume or produce *tensor_name*."""
        return self._clone([
            n for n in self.nodes
            if tensor_name in n.input or tensor_name in n.output
        ])

    def find_by_param_name(self, name: str, *, exact: bool = False) -> "ONNXQuery":
        """Filter to nodes whose initializer inputs match *name*."""
        def _match(p: str) -> bool:
            return p == name if exact else name.lower() in p.lower()
        return self._clone([
            n for n in self.nodes if any(_match(p) for p in self._params(n))
        ])

    def find_by_attribute(self, attr_name: str, value: Any = None) -> "ONNXQuery":
        """
        Filter to nodes that have *attr_name*, optionally matching *value*.
        *value* may be a plain value (equality) or a callable predicate.
        """
        def _has(node: NodeProto) -> bool:
            for attr in node.attribute:
                if attr.name != attr_name:
                    continue
                if value is None:
                    return True
                av = _attr_value(attr)
                return value(av) if callable(value) else av == value
            return False
        return self._clone([n for n in self.nodes if _has(n)])

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def children(self) -> "ONNXQuery":
        """Nodes that consume any output produced by the current set."""
        outputs = {out for n in self.nodes for out in n.output}
        return self._clone([
            n for n in self.all_nodes if any(i in outputs for i in n.input)
        ])

    def parents(self) -> "ONNXQuery":
        """Nodes that produce any input consumed by the current set."""
        inputs     = {inp for n in self.nodes for inp in n.input}
        output_map = self._build_output_map()
        seen: Set[str] = set()
        result: List[NodeProto] = []
        for inp in inputs:
            p = output_map.get(inp)
            if p and p.name not in seen:
                seen.add(p.name)
                result.append(p)
        return self._clone(result)

    def ancestors(self, max_depth: int = 100) -> "ONNXQuery":
        """All transitive parents up to *max_depth* hops (current nodes excluded)."""
        output_map = self._build_output_map()
        visited: Set[str] = set()
        frontier  = list(self.nodes)
        result:   List[NodeProto] = []
        for _ in range(max_depth):
            nxt: List[NodeProto] = []
            for n in frontier:
                for inp in n.input:
                    p = output_map.get(inp)
                    if p and p.name not in visited:
                        visited.add(p.name)
                        result.append(p)
                        nxt.append(p)
            if not nxt:
                break
            frontier = nxt
        return self._clone(result)

    def descendants(self, max_depth: int = 100) -> "ONNXQuery":
        """All transitive children up to *max_depth* hops (current nodes excluded)."""
        input_map: Dict[str, List[NodeProto]] = {}
        for n in self.all_nodes:
            for inp in n.input:
                input_map.setdefault(inp, []).append(n)
        visited: Set[str] = set()
        frontier  = list(self.nodes)
        result:   List[NodeProto] = []
        for _ in range(max_depth):
            nxt: List[NodeProto] = []
            for n in frontier:
                for out in n.output:
                    for child in input_map.get(out, []):
                        if child.name not in visited:
                            visited.add(child.name)
                            result.append(child)
                            nxt.append(child)
            if not nxt:
                break
            frontier = nxt
        return self._clone(result)

    # ------------------------------------------------------------------
    # Entry / exit
    # ------------------------------------------------------------------

    @property
    def entry_nodes(self) -> "ONNXQuery":
        """Nodes whose inputs include at least one graph-level input."""
        return self._clone([
            n for n in self.all_nodes
            if any(i in self.graph_inputs for i in n.input)
        ])

    @property
    def output_nodes(self) -> "ONNXQuery":
        """Nodes whose outputs include at least one graph-level output."""
        return self._clone([
            n for n in self.all_nodes
            if any(o in self.graph_outputs for o in n.output)
        ])

    # ------------------------------------------------------------------
    # Param access
    # ------------------------------------------------------------------

    def has_params(self) -> "ONNXQuery":
        """Filter to nodes that have at least one weight tensor."""
        return self._clone([n for n in self.nodes if self._params(n)])

    def tensor(self, name: Optional[str] = None) -> Any:
        """
        Return weight tensor(s):
        - *name* given → single array or None
        - 1 node selected → {param_name: array}
        - multiple nodes → {node_name: {param_name: array}}
        """
        if name is not None:
            return self.tensor_map.get(name)
        if len(self.nodes) == 1:
            return self._params(self.nodes[0])
        return {n.name: p for n in self.nodes if (p := self._params(n))}

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------

    def matches(self, pattern: Pattern) -> "ONNXQuery":
        """
        Return the subset of nodes that are the start of a subgraph
        matching *pattern*.  Shares the output→node map across all tests.
        """
        output_map = self._build_output_map()
        shim = _GraphShim(self.all_nodes, self.tensor_map, self.shape_info)
        matched: List[NodeProto] = []
        for node in self.nodes:
            det = PatternDetector(shim, start_node=node)
            det._output_to_node = output_map
            if det.match(pattern) is not None:
                matched.append(node)
        return self._clone(matched)

    def match_results(self, pattern: Pattern) -> List[MatchResult]:
        """
        Like :meth:`matches` but returns a :class:`MatchResult` for each
        matching node, giving access to bindings, visited nodes, etc.
        """
        output_map = self._build_output_map()
        shim = _GraphShim(self.all_nodes, self.tensor_map, self.shape_info)
        results: List[MatchResult] = []
        for node in self.nodes:
            det = PatternDetector(shim, start_node=node)
            det._output_to_node = output_map
            r = det.match(pattern)
            if r is not None:
                # Attach tensor_map so as_query() works correctly
                r._tensor_map = self.tensor_map  # type: ignore[attr-defined]
                results.append(r)
        return results

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    def topological_sort(self) -> "ONNXQuery":
        """
        Return a new ONNXQuery with the selected nodes in topological order
        (Kahn's algorithm).  Raises ValueError if a cycle is detected.
        """
        nodes  = self.nodes
        name_set = {n.name for n in nodes}
        output_map = self._build_output_map()

        # in-degree relative to the selected subgraph only
        in_deg: Dict[str, int] = {n.name: 0 for n in nodes}
        for n in nodes:
            for inp in n.input:
                p = output_map.get(inp)
                if p and p.name in name_set:
                    in_deg[n.name] += 1

        queue = [n for n in nodes if in_deg[n.name] == 0]
        result: List[NodeProto] = []
        while queue:
            cur = queue.pop(0)
            result.append(cur)
            for out in cur.output:
                for n in nodes:
                    if out in n.input and n.name in name_set:
                        in_deg[n.name] -= 1
                        if in_deg[n.name] == 0:
                            queue.append(n)

        if len(result) != len(nodes):
            raise ValueError("Cycle detected in selected subgraph — cannot topologically sort.")
        return self._clone(result)

    def is_topologically_sorted(self) -> bool:
        """Return True if the selected nodes are already in topological order."""
        try:
            sorted_q = self.topological_sort()
        except ValueError:
            return False
        names_orig   = [n.name for n in self.nodes]
        names_sorted = [n.name for n in sorted_q.nodes]
        return names_orig == names_sorted

    # ------------------------------------------------------------------
    # Functional transform
    # ------------------------------------------------------------------

    def apply(self, fn: Callable[[NodeProto, TensorMap], Any]) -> "ONNXQuery":
        """
        Call ``fn(node, params)`` on every selected node and return *self*
        unchanged (for chaining side-effectful operations).

        *params* is the dict of weight tensors for that node (may be empty).

        Example::

            parser.find().find_by_op_type("Conv").apply(
                lambda n, p: print(n.name, p["weight"].shape)
            )
        """
        for node in self.nodes:
            fn(node, self._params(node))
        return self

    # ------------------------------------------------------------------
    # Set operations
    # ------------------------------------------------------------------

    def union(self, other: "ONNXQuery") -> "ONNXQuery":
        seen = {n.name for n in self.nodes}
        return self._clone(self.nodes + [n for n in other.nodes if n.name not in seen])

    def intersection(self, other: "ONNXQuery") -> "ONNXQuery":
        other_names = {n.name for n in other.nodes}
        return self._clone([n for n in self.nodes if n.name in other_names])

    def difference(self, other: "ONNXQuery") -> "ONNXQuery":
        other_names = {n.name for n in other.nodes}
        return self._clone([n for n in self.nodes if n.name not in other_names])

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dot(self, *, show_params: bool = False) -> str:
        """
        Return a Graphviz DOT string visualising the selected nodes.

        Parameters
        ----------
        show_params : bool
            When True, include initializer names as leaf nodes.

        Example::

            dot_src = parser.find().to_dot()
            # paste into https://dreampuf.github.io/GraphvizOnline/
        """
        lines = ["digraph onnx_subgraph {", '  rankdir="LR";',
                 '  node [shape=box fontname="monospace"];']
        output_map = self._build_output_map()
        node_names = {n.name for n in self.nodes}

        for n in self.nodes:
            label = f"{n.op_type}\\n{n.name}"
            lines.append(f'  "{n.name}" [label="{label}"];')

        for n in self.nodes:
            for inp in n.input:
                if not inp:
                    continue
                if inp in self.tensor_map:
                    if show_params:
                        arr   = self.tensor_map[inp]
                        label = f"param\\n{inp}\\n{list(arr.shape)}"
                        lines.append(f'  "{inp}" [label="{label}" shape=ellipse];')
                        lines.append(f'  "{inp}" -> "{n.name}";')
                else:
                    parent = output_map.get(inp)
                    if parent and parent.name in node_names:
                        lines.append(f'  "{parent.name}" -> "{n.name}";')

        lines.append("}")
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        """
        Return a pandas DataFrame with one row per selected node.

        Columns: name, op_type, num_inputs, num_outputs, num_params,
                 param_shapes, attributes.

        Raises ImportError if pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for to_dataframe()") from e

        rows = []
        for n in self.nodes:
            params = self._params(n)
            attrs  = {a.name: _attr_value(a) for a in n.attribute}
            rows.append({
                "name":         n.name,
                "op_type":      n.op_type,
                "num_inputs":   len(n.input),
                "num_outputs":  len(n.output),
                "num_params":   len(params),
                "param_shapes": {k: list(v.shape) for k, v in params.items()},
                "attributes":   attrs,
            })
        return pd.DataFrame(rows)

    def to_onnx(self, path: str) -> None:
        """
        Save the selected subgraph as a minimal standalone ``.onnx`` file.

        Inputs and outputs of the subgraph are inferred automatically:
        a tensor is an **input** if it is consumed but not produced within
        the selection; it is an **output** if it is produced but not consumed
        within the selection (or is a graph-level output).

        Raises ImportError if onnx is not installed.
        """
        import onnx
        from onnx import helper, TensorProto

        node_names   = {n.name for n in self.nodes}
        produced: Set[str] = {out for n in self.nodes for out in n.output if out}
        consumed: Set[str] = {inp for n in self.nodes for inp in n.input  if inp}

        # External inputs: consumed but not produced within the subgraph
        # and not an initializer
        external_inputs  = consumed - produced - set(self.tensor_map.keys())
        # External outputs: produced but not consumed within the subgraph
        external_outputs = produced - consumed

        def _make_vi(name: str) -> Any:
            return helper.make_tensor_value_info(name, TensorProto.FLOAT, None)

        graph = helper.make_graph(
            nodes       = list(self.nodes),
            name        = "subgraph",
            inputs      = [_make_vi(n) for n in sorted(external_inputs)],
            outputs     = [_make_vi(n) for n in sorted(external_outputs)],
            initializer = [
                onnx.numpy_helper.from_array(v, name=k)
                for k, v in self.tensor_map.items()
                if k in consumed
            ],
        )
        model = helper.make_model(graph)
        onnx.save(model, path)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def single_node(self) -> NodeProto:
        if len(self.nodes) != 1:
            raise ValueError(f"Expected exactly 1 node, got {len(self.nodes)}")
        return self.nodes[0]

    @property
    def single_tensor(self) -> np.ndarray:
        params = self._params(self.single_node)
        if not params:
            raise ValueError("No weight tensor for this node")
        if len(params) != 1:
            raise ValueError(f"Multiple tensors found: {list(params.keys())}")
        return next(iter(params.values()))

    def first(self) -> Optional[NodeProto]:
        return self.nodes[0] if self.nodes else None

    def last(self) -> Optional[NodeProto]:
        return self.nodes[-1] if self.nodes else None

    def count(self) -> int:
        return len(self.nodes)

    def is_empty(self) -> bool:
        return not self.nodes

    def op_types(self) -> List[str]:
        """Deduplicated list of op types in selection order."""
        seen: Set[str] = set()
        result = []
        for n in self.nodes:
            if n.op_type not in seen:
                seen.add(n.op_type)
                result.append(n.op_type)
        return result

    # ------------------------------------------------------------------
    # Iteration / indexing
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[NodeProto]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __bool__(self) -> bool:
        return bool(self.nodes)

    def __getitem__(self, idx: Union[int, slice]) -> "ONNXQuery":
        nodes = self.nodes[idx] if isinstance(idx, slice) else [self.nodes[idx]]
        return self._clone(nodes)

    def __repr__(self) -> str:
        if not self.nodes:
            return "ONNXQuery(empty)"
        lines = [f"ONNXQuery: {len(self.nodes)} node(s)"]
        for i, n in enumerate(self.nodes[:15]):
            params = self._params(n)
            pstr   = ", ".join(f"{k}:{list(v.shape)}" for k, v in params.items()) or "—"
            lines.append(f"  [{i:>3}] {n.op_type:<16} {n.name}")
            lines.append(f"         params: {pstr}")
        if len(self.nodes) > 15:
            lines.append(f"  … ({len(self.nodes) - 15} more)")
        return "\n".join(lines)
