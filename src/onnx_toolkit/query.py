"""
onnx_toolkit.query
===================
ONNXQuery: a lazy, chainable view over a subset of ONNX graph nodes.

Every filter / traversal method returns a *new* ONNXQuery so calls can be
chained freely without mutating the receiver::

    parser.find()
          .find_by_op_type("Conv")
          .has_params()
          .children()
          .find_by_op_type("Relu")

Logging
-------
  onnx_toolkit.query  – filter & traversal steps
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Set, Union

import numpy as np
from onnx.onnx_pb import NodeProto

from ._types import TensorMap
from ._utils import _GraphShim, _attr_value
from .pattern import Pattern, PatternDetector

logger_query = logging.getLogger("onnx_toolkit.query")


class ONNXQuery:
    """
    A lazy, chainable view over a subset of ONNX graph nodes.

    Parameters
    ----------
    nodes : list[NodeProto]
        The currently selected nodes.
    tensor_map : TensorMap
        Weight tensors keyed by initializer name.
    all_nodes : list[NodeProto]
        Every node in the graph (used for traversal).
    graph_inputs : set[str]
        Names of graph-level inputs.
    graph_outputs : set[str]
        Names of graph-level outputs.
    """

    def __init__(
        self,
        nodes: List[NodeProto],
        tensor_map: TensorMap,
        all_nodes: List[NodeProto],
        graph_inputs: Optional[Set[str]] = None,
        graph_outputs: Optional[Set[str]] = None,
    ) -> None:
        self.nodes: List[NodeProto] = nodes
        self.tensor_map: TensorMap = tensor_map
        self.all_nodes: List[NodeProto] = all_nodes
        self.graph_inputs: Set[str] = graph_inputs or set()
        self.graph_outputs: Set[str] = graph_outputs or set()
        self._output_to_node: Optional[Dict[str, NodeProto]] = None

        logger_query.debug(
            "ONNXQuery created: %d selected node(s) out of %d total",
            len(self.nodes), len(self.all_nodes),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_output_map(self) -> Dict[str, NodeProto]:
        if self._output_to_node is None:
            logger_query.debug(
                "Building output→node map for %d nodes", len(self.all_nodes)
            )
            self._output_to_node = {}
            for node in self.all_nodes:
                for out in node.output:
                    self._output_to_node[out] = node
            logger_query.debug(
                "Output map built: %d entries", len(self._output_to_node)
            )
        else:
            logger_query.debug(
                "Reusing cached output→node map (%d entries)",
                len(self._output_to_node),
            )
        return self._output_to_node

    def _clone(self, nodes: List[NodeProto]) -> "ONNXQuery":
        logger_query.debug("Cloning ONNXQuery with %d node(s)", len(nodes))
        q = ONNXQuery(
            nodes,
            self.tensor_map,
            self.all_nodes,
            self.graph_inputs,
            self.graph_outputs,
        )
        q._output_to_node = self._output_to_node  # share cached map
        return q

    def _extract_params(self, node: NodeProto) -> TensorMap:
        params = {
            inp: self.tensor_map[inp]
            for inp in node.input
            if inp in self.tensor_map
        }
        logger_query.debug(
            "_extract_params(%s [%s]): found %d weight tensor(s): %s",
            node.name, node.op_type, len(params), list(params.keys()),
        )
        return params

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def find_by_op_type(self, op_type: str) -> "ONNXQuery":
        """Return nodes whose op_type equals *op_type* (case-sensitive)."""
        logger_query.debug(
            "find_by_op_type(%r) on %d node(s)", op_type, len(self.nodes)
        )
        result = self._clone([n for n in self.nodes if n.op_type == op_type])
        logger_query.debug(
            "find_by_op_type(%r) → %d match(es)", op_type, result.count()
        )
        return result

    def find_by_name(self, name: str, *, exact: bool = False) -> "ONNXQuery":
        """
        Return nodes matching *name*.

        Parameters
        ----------
        name  : str   Substring (default) or exact name to search.
        exact : bool  When True use equality; otherwise case-insensitive substring.
        """
        logger_query.debug(
            "find_by_name(%r, exact=%s) on %d node(s)", name, exact, len(self.nodes)
        )
        if exact:
            filtered = [n for n in self.nodes if n.name == name]
        else:
            low = name.lower()
            filtered = [n for n in self.nodes if low in n.name.lower()]
        logger_query.debug("find_by_name(%r) → %d match(es)", name, len(filtered))
        return self._clone(filtered)

    def find_by_tensor(self, tensor_name: str) -> "ONNXQuery":
        """Return nodes that consume or produce a tensor named *tensor_name*."""
        logger_query.debug(
            "find_by_tensor(%r) on %d node(s)", tensor_name, len(self.nodes)
        )
        result = self._clone([
            n for n in self.nodes
            if tensor_name in n.input or tensor_name in n.output
        ])
        logger_query.debug(
            "find_by_tensor(%r) → %d match(es)", tensor_name, result.count()
        )
        return result

    def find_by_param_name(self, name: str, *, exact: bool = False) -> "ONNXQuery":
        """
        Return nodes whose weight (initializer) inputs match *name*.

        Parameters
        ----------
        name  : str   Substring (default) or exact weight name.
        exact : bool  When True use equality; otherwise case-insensitive substring.
        """
        logger_query.debug(
            "find_by_param_name(%r, exact=%s) on %d node(s)",
            name, exact, len(self.nodes),
        )

        def _match(param_name: str) -> bool:
            return param_name == name if exact else name.lower() in param_name.lower()

        result = self._clone([
            n for n in self.nodes
            if any(_match(p) for p in self._extract_params(n))
        ])
        logger_query.debug(
            "find_by_param_name(%r) → %d match(es)", name, result.count()
        )
        return result

    def find_by_attribute(self, attr_name: str, value: Any = None) -> "ONNXQuery":
        """
        Return nodes that have an attribute named *attr_name*, optionally
        matching *value*.

        Parameters
        ----------
        attr_name : str  ONNX attribute name (e.g. ``"group"``, ``"dilations"``).
        value     : Any  When provided, the attribute value must equal this.
        """
        logger_query.debug(
            "find_by_attribute(%r, value=%r) on %d node(s)",
            attr_name, value, len(self.nodes),
        )

        def _has_attr(node: NodeProto) -> bool:
            for attr in node.attribute:
                if attr.name == attr_name:
                    if value is None:
                        logger_query.debug(
                            "  Node %r has attribute %r (value check skipped)",
                            node.name, attr_name,
                        )
                        return True
                    av = _attr_value(attr)
                    match = av == value
                    logger_query.debug(
                        "  Node %r attr %r: found value=%r, expected=%r → %s",
                        node.name, attr_name, av, value,
                        "MATCH" if match else "no match",
                    )
                    return match
            logger_query.debug(
                "  Node %r: attribute %r not present", node.name, attr_name
            )
            return False

        result = self._clone([n for n in self.nodes if _has_attr(n)])
        logger_query.debug(
            "find_by_attribute(%r) → %d match(es)", attr_name, result.count()
        )
        return result

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def children(self) -> "ONNXQuery":
        """Return nodes that consume any output produced by the current set."""
        logger_query.debug("children() from %d node(s)", len(self.nodes))
        outputs: Set[str] = {out for n in self.nodes for out in n.output}
        logger_query.debug("  Produced outputs to match against: %s", outputs)
        result = self._clone([
            node for node in self.all_nodes
            if any(inp in outputs for inp in node.input)
        ])
        logger_query.debug("children() → %d child node(s)", result.count())
        return result

    def parents(self) -> "ONNXQuery":
        """Return nodes that produce any input consumed by the current set."""
        logger_query.debug("parents() from %d node(s)", len(self.nodes))
        inputs: Set[str] = {inp for n in self.nodes for inp in n.input}
        logger_query.debug("  Consumed inputs to trace back: %s", inputs)
        output_map = self._build_output_map()
        seen: Set[str] = set()
        result = []
        for inp in inputs:
            if inp in output_map:
                n = output_map[inp]
                if n.name not in seen:
                    seen.add(n.name)
                    result.append(n)
                    logger_query.debug(
                        "  Input %r produced by node %r [%s]",
                        inp, n.name, n.op_type,
                    )
        logger_query.debug("parents() → %d parent node(s)", len(result))
        return self._clone(result)

    def ancestors(self, max_depth: int = 100) -> "ONNXQuery":
        """
        Return all transitive parent nodes up to *max_depth* hops away.
        The current nodes themselves are **not** included.
        """
        logger_query.debug(
            "ancestors(max_depth=%d) from %d node(s)", max_depth, len(self.nodes)
        )
        visited: Set[str] = set()
        frontier = list(self.nodes)
        result: List[NodeProto] = []
        output_map = self._build_output_map()

        for depth in range(max_depth):
            next_frontier: List[NodeProto] = []
            for n in frontier:
                for inp in n.input:
                    parent = output_map.get(inp)
                    if parent and parent.name not in visited:
                        visited.add(parent.name)
                        result.append(parent)
                        next_frontier.append(parent)
                        logger_query.debug(
                            "  depth=%d: found ancestor %r [%s] via input %r",
                            depth + 1, parent.name, parent.op_type, inp,
                        )
            if not next_frontier:
                logger_query.debug(
                    "  Frontier exhausted at depth=%d, stopping", depth + 1
                )
                break
            frontier = next_frontier

        logger_query.debug("ancestors() → %d ancestor node(s)", len(result))
        return self._clone(result)

    def descendants(self, max_depth: int = 100) -> "ONNXQuery":
        """
        Return all transitive child nodes up to *max_depth* hops away.
        The current nodes themselves are **not** included.
        """
        logger_query.debug(
            "descendants(max_depth=%d) from %d node(s)", max_depth, len(self.nodes)
        )
        visited: Set[str] = set()
        input_map: Dict[str, List[NodeProto]] = {}
        for node in self.all_nodes:
            for inp in node.input:
                input_map.setdefault(inp, []).append(node)

        frontier = list(self.nodes)
        result: List[NodeProto] = []

        for depth in range(max_depth):
            next_frontier: List[NodeProto] = []
            for n in frontier:
                for out in n.output:
                    for child in input_map.get(out, []):
                        if child.name not in visited:
                            visited.add(child.name)
                            result.append(child)
                            next_frontier.append(child)
                            logger_query.debug(
                                "  depth=%d: found descendant %r [%s] via output %r",
                                depth + 1, child.name, child.op_type, out,
                            )
            if not next_frontier:
                logger_query.debug(
                    "  Frontier exhausted at depth=%d, stopping", depth + 1
                )
                break
            frontier = next_frontier

        logger_query.debug("descendants() → %d descendant node(s)", len(result))
        return self._clone(result)

    # ------------------------------------------------------------------
    # Entry / exit shortcuts
    # ------------------------------------------------------------------

    @property
    def entry_nodes(self) -> "ONNXQuery":
        """Nodes whose inputs include at least one graph-level input."""
        logger_query.debug(
            "entry_nodes: scanning %d node(s) for graph inputs %s",
            len(self.all_nodes), self.graph_inputs,
        )
        result = self._clone([
            n for n in self.all_nodes
            if any(inp in self.graph_inputs for inp in n.input)
        ])
        logger_query.debug("entry_nodes → %d node(s)", result.count())
        return result

    @property
    def output_nodes(self) -> "ONNXQuery":
        """Nodes whose outputs include at least one graph-level output."""
        logger_query.debug(
            "output_nodes: scanning %d node(s) for graph outputs %s",
            len(self.all_nodes), self.graph_outputs,
        )
        result = self._clone([
            n for n in self.all_nodes
            if any(out in self.graph_outputs for out in n.output)
        ])
        logger_query.debug("output_nodes → %d node(s)", result.count())
        return result

    # ------------------------------------------------------------------
    # Param access
    # ------------------------------------------------------------------

    def has_params(self) -> "ONNXQuery":
        """Filter to nodes that have at least one weight tensor."""
        logger_query.debug("has_params() on %d node(s)", len(self.nodes))
        result = self._clone([n for n in self.nodes if self._extract_params(n)])
        logger_query.debug("has_params() → %d node(s) with weights", result.count())
        return result

    def tensor(self, name: Optional[str] = None) -> Any:
        """
        Return weight tensor(s) for the selected nodes.

        - If *name* is given: return that tensor directly (or ``None``).
        - If exactly one node is selected: return ``{param_name: array, ...}``.
        - Otherwise: return ``{node_name: {param_name: array}, ...}``.
        """
        logger_query.debug(
            "tensor(name=%r) on %d selected node(s)", name, len(self.nodes)
        )
        if name is not None:
            value = self.tensor_map.get(name)
            if value is not None:
                logger_query.debug(
                    "  tensor(%r): found, shape=%s", name, value.shape
                )
            else:
                logger_query.debug(
                    "  tensor(%r): NOT found in tensor_map", name
                )
            return value

        if len(self.nodes) == 1:
            params = self._extract_params(self.nodes[0])
            logger_query.debug(
                "  Single-node tensor(): returning %d param(s)", len(params)
            )
            return params

        result = {
            n.name: params
            for n in self.nodes
            if (params := self._extract_params(n))
        }
        logger_query.debug(
            "  Multi-node tensor(): returning params for %d node(s)", len(result)
        )
        return result

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------

    def matches(self, pattern: Pattern) -> "ONNXQuery":
        """
        Return the subset of nodes from the current selection that are
        the *start* node of a subgraph matching *pattern*.

        Uses a single shared output→node map across all candidate nodes
        to avoid O(n²) rebuilds.
        """
        logger_query.debug(
            "matches(pattern) on %d candidate node(s)", len(self.nodes)
        )
        shim = _GraphShim(self.all_nodes, self.tensor_map)
        output_map = self._build_output_map()
        matched: List[NodeProto] = []

        for node in self.nodes:
            logger_query.debug(
                "  Testing node %r [%s] against pattern", node.name, node.op_type
            )
            detector = PatternDetector(shim, start_node=node)
            detector._output_to_node = output_map  # inject shared map
            if detector.match(pattern) is not None:
                logger_query.debug(
                    "  Node %r [%s] → MATCHED", node.name, node.op_type
                )
                matched.append(node)
            else:
                logger_query.debug(
                    "  Node %r [%s] → no match", node.name, node.op_type
                )

        logger_query.debug("matches() → %d matching node(s)", len(matched))
        return self._clone(matched)

    # ------------------------------------------------------------------
    # Set operations
    # ------------------------------------------------------------------

    def union(self, other: "ONNXQuery") -> "ONNXQuery":
        """Return nodes present in *self* **or** *other* (deduplicated)."""
        logger_query.debug(
            "union(): self=%d, other=%d", len(self.nodes), len(other.nodes)
        )
        seen: Set[str] = {n.name for n in self.nodes}
        extra = [n for n in other.nodes if n.name not in seen]
        result = self._clone(self.nodes + extra)
        logger_query.debug("union() → %d node(s)", result.count())
        return result

    def intersection(self, other: "ONNXQuery") -> "ONNXQuery":
        """Return nodes present in *self* **and** *other*."""
        logger_query.debug(
            "intersection(): self=%d, other=%d", len(self.nodes), len(other.nodes)
        )
        other_names: Set[str] = {n.name for n in other.nodes}
        result = self._clone([n for n in self.nodes if n.name in other_names])
        logger_query.debug("intersection() → %d node(s)", result.count())
        return result

    def difference(self, other: "ONNXQuery") -> "ONNXQuery":
        """Return nodes present in *self* but **not** *other*."""
        logger_query.debug(
            "difference(): self=%d, other=%d", len(self.nodes), len(other.nodes)
        )
        other_names: Set[str] = {n.name for n in other.nodes}
        result = self._clone([n for n in self.nodes if n.name not in other_names])
        logger_query.debug("difference() → %d node(s)", result.count())
        return result

    # ------------------------------------------------------------------
    # Helpers / accessors
    # ------------------------------------------------------------------

    @property
    def single_node(self) -> NodeProto:
        if len(self.nodes) != 1:
            logger_query.error(
                "single_node: expected 1 node, got %d", len(self.nodes)
            )
            raise ValueError(f"Expected exactly 1 node, got {len(self.nodes)}")
        logger_query.debug(
            "single_node → %r [%s]", self.nodes[0].name, self.nodes[0].op_type
        )
        return self.nodes[0]

    @property
    def single_tensor(self) -> np.ndarray:
        params = self._extract_params(self.single_node)
        if not params:
            logger_query.error("single_tensor: no weight tensor found")
            raise ValueError("No weight tensor for this node")
        if len(params) != 1:
            logger_query.error(
                "single_tensor: multiple tensors found: %s", list(params.keys())
            )
            raise ValueError(f"Multiple tensors found: {list(params.keys())}")
        tensor = next(iter(params.values()))
        logger_query.debug("single_tensor → shape=%s", tensor.shape)
        return tensor

    def first(self) -> Optional[NodeProto]:
        """Return the first node or ``None`` if empty."""
        node = self.nodes[0] if self.nodes else None
        logger_query.debug(
            "first() → %s",
            f"{node.name!r} [{node.op_type}]" if node else "None",
        )
        return node

    def last(self) -> Optional[NodeProto]:
        """Return the last node or ``None`` if empty."""
        node = self.nodes[-1] if self.nodes else None
        logger_query.debug(
            "last() → %s",
            f"{node.name!r} [{node.op_type}]" if node else "None",
        )
        return node

    def count(self) -> int:
        """Number of nodes in the current selection."""
        return len(self.nodes)

    def is_empty(self) -> bool:
        return len(self.nodes) == 0

    def op_types(self) -> List[str]:
        """Return a deduplicated list of op types in the selection."""
        seen: Set[str] = set()
        result = []
        for n in self.nodes:
            if n.op_type not in seen:
                seen.add(n.op_type)
                result.append(n.op_type)
        logger_query.debug("op_types() → %s", result)
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
        if isinstance(idx, slice):
            result = self._clone(self.nodes[idx])
            logger_query.debug("__getitem__(slice) → %d node(s)", result.count())
            return result
        result = self._clone([self.nodes[idx]])
        logger_query.debug(
            "__getitem__(%d) → %r [%s]",
            idx, result.nodes[0].name, result.nodes[0].op_type,
        )
        return result

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self.nodes:
            return "ONNXQuery(empty)"

        lines = [f"ONNXQuery: {len(self.nodes)} node(s)"]
        for i, n in enumerate(self.nodes[:15]):
            params = self._extract_params(n)
            param_str = (
                ", ".join(f"{k}:{v.shape}" for k, v in params.items())
                if params else "None"
            )
            lines.append(f"  [{i:>3}] {n.op_type:<16} {n.name}")
            lines.append(f"         params: {param_str}")

        if len(self.nodes) > 15:
            lines.append(f"  ... ({len(self.nodes) - 15} more nodes)")

        return "\n".join(lines)
