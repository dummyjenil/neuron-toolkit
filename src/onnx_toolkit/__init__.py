"""
onnx_toolkit
================
Unified ONNX parsing and pattern-matching library.

Provides two tightly integrated components:
  - ONNXParser / ONNXQuery  : load a model and query its graph with a fluent API
  - Pattern / Pattern.detect : describe structural patterns and match them against
                               the graph using DFS with commutativity support

The two components are coupled: Pattern.detect can be seeded directly from an
ONNXQuery result, and ONNXQuery exposes a .matches(pattern) convenience method
that delegates to Pattern.detect internally.

Logging
-------
This module uses Python's standard ``logging`` library under the logger name
``"onnx_toolkit"``.  To enable debug output in your script::

    import logging
    logging.basicConfig(level=logging.DEBUG)

Or to only enable it for this module::

    logging.getLogger("onnx_toolkit").setLevel(logging.DEBUG)
    logging.getLogger("onnx_toolkit").addHandler(logging.StreamHandler())

Sub-loggers
-----------
  onnx_toolkit               - top-level / parser messages
  onnx_toolkit.query         - ONNXQuery filter & traversal steps
  onnx_toolkit.pattern       - Pattern construction messages
  onnx_toolkit.detect        - Pattern.detect DFS walk & decisions
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, Iterator, List, Optional, Set, Union

import numpy as np
import onnx
from onnx import ModelProto, numpy_helper
from onnx.onnx_pb import NodeProto


# ---------------------------------------------------------------------------
# Loggers
# ---------------------------------------------------------------------------
logger         = logging.getLogger("onnx_toolkit")
logger_query   = logging.getLogger("onnx_toolkit.query")
logger_pattern = logging.getLogger("onnx_toolkit.pattern")
logger_detect  = logging.getLogger("onnx_toolkit.detect")


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
TensorMap = Dict[str, np.ndarray]


# ===========================================================================
# ONNXQuery
# ===========================================================================

class ONNXQuery:
    """
    A lazy, chainable view over a subset of ONNX graph nodes.

    Every filter / traversal method returns a *new* ONNXQuery so calls can be
    chained freely without mutating the receiver:

        parser.find()
              .find_by_op_type("Conv")
              .has_params()
              .children()
              .find_by_op_type("Relu")

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

        # Lazily built lookup: output_name -> node  (shared across queries)
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
            logger_query.debug("Building output→node map for %d nodes", len(self.all_nodes))
            self._output_to_node = {}
            for node in self.all_nodes:
                for out in node.output:
                    self._output_to_node[out] = node
            logger_query.debug("Output map built: %d entries", len(self._output_to_node))
        else:
            logger_query.debug("Reusing cached output→node map (%d entries)", len(self._output_to_node))
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
        # Share the already-built lookup to avoid O(n) rebuilds on every call
        q._output_to_node = self._output_to_node
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
        logger_query.debug("find_by_tensor(%r) → %d match(es)", tensor_name, result.count())
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
            "find_by_param_name(%r, exact=%s) on %d node(s)", name, exact, len(self.nodes)
        )

        def _match(param_name: str) -> bool:
            return param_name == name if exact else name.lower() in param_name.lower()

        result = self._clone([
            n for n in self.nodes
            if any(_match(p) for p in self._extract_params(n))
        ])
        logger_query.debug("find_by_param_name(%r) → %d match(es)", name, result.count())
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
                    # Extract typed value
                    av = _attr_value(attr)
                    match = av == value
                    logger_query.debug(
                        "  Node %r attr %r: found value=%r, expected=%r → %s",
                        node.name, attr_name, av, value, "MATCH" if match else "no match",
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
                        "  Input %r produced by node %r [%s]", inp, n.name, n.op_type
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
        # Build input map: input_name -> list[node]
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
                logger_query.debug("  tensor(%r): NOT found in tensor_map", name)
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
    # Pattern matching (integration with Pattern.detect)
    # ------------------------------------------------------------------

    def matches(self, pattern: "Pattern") -> "ONNXQuery":
        """
        Return the subset of nodes from the current selection that are
        the *start* node of a subgraph matching *pattern*.
        """
        logger_query.debug(
            "matches(pattern) on %d candidate node(s)", len(self.nodes)
        )
        shim = _GraphShim(self.all_nodes, self.tensor_map)
        output_map = self._build_output_map()  # built once, shared across all iterations
        matched: List[NodeProto] = []

        for node in self.nodes:
            logger_query.debug(
                "  Testing node %r [%s] against pattern", node.name, node.op_type
            )
            detector = Pattern.detect(shim, start_node=node)
            detector._output_to_node = output_map   # inject shared map, no rebuild
            if detector.match(pattern):
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
        logger_query.debug("single_node → %r [%s]", self.nodes[0].name, self.nodes[0].op_type)
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


# ===========================================================================
# ONNXParser
# ===========================================================================

class ONNXParser:
    """
    Load an ONNX model and expose it for querying.

    Parameters
    ----------
    onnx_path : str  Path to the ``.onnx`` file.

    Example
    -------
    >>> parser = ONNXParser("model.onnx")
    >>> convs = parser.find().find_by_op_type("Conv").has_params()
    >>> print(convs)
    """

    def __init__(self, onnx_path: str) -> None:
        logger.info("Loading ONNX model from %r", onnx_path)
        self.model: ModelProto = onnx.load(onnx_path)
        self.nodes: List[NodeProto] = list(self.model.graph.node)
        self.tensor_map: TensorMap = {
            t.name: numpy_helper.to_array(t)
            for t in self.model.graph.initializer
        }
        self.graph_inputs: Set[str] = {i.name for i in self.model.graph.input}
        self.graph_outputs: Set[str] = {o.name for o in self.model.graph.output}
        logger.info(
            "Model loaded: %d nodes, %d tensors, %d graph input(s), %d graph output(s)",
            len(self.nodes),
            len(self.tensor_map),
            len(self.graph_inputs),
            len(self.graph_outputs),
        )
        logger.debug("Graph inputs : %s", self.graph_inputs)
        logger.debug("Graph outputs: %s", self.graph_outputs)

    # ------------------------------------------------------------------

    def find(self) -> ONNXQuery:
        """Return an ONNXQuery over **all** nodes in the graph."""
        logger.debug("find(): returning ONNXQuery over all %d nodes", len(self.nodes))
        return ONNXQuery(
            list(self.nodes),
            self.tensor_map,
            self.nodes,
            self.graph_inputs,
            self.graph_outputs,
        )

    def pattern_detect(
        self,
        pattern: "Pattern",
        start_node: Optional[Union[str, NodeProto]] = None,
        end_node: Optional[Union[str, NodeProto]] = None,
    ):
        """
        Convenience shortcut: create a Pattern.detect bound to this model.

        Parameters
        ----------
        pattern    : Pattern   The pattern to check.
        start_node : optional  Seed node (name or NodeProto).
        end_node   : optional  Stop node (name or NodeProto).
        """
        logger.debug(
            "pattern_detect(start_node=%r, end_node=%r)",
            start_node if isinstance(start_node, str) else getattr(start_node, "name", start_node),
            end_node   if isinstance(end_node,   str) else getattr(end_node,   "name", end_node),
        )
        detector = Pattern.detect(self.model, start_node=start_node, end_node=end_node)
        result = detector.match(pattern)
        logger.debug("pattern_detect() → %s", result)
        return result

    def summary(self) -> str:
        """Return a human-readable summary of the model."""
        from collections import Counter
        op_counts = Counter(n.op_type for n in self.nodes)
        lines = [
            f"ONNX model summary",
            f"  Nodes       : {len(self.nodes)}",
            f"  Tensors     : {len(self.tensor_map)}",
            f"  Graph inputs: {len(self.graph_inputs)}",
            f"  Graph outputs:{len(self.graph_outputs)}",
            f"  Op types ({len(op_counts)}):",
        ]
        for op, cnt in op_counts.most_common():
            lines.append(f"    {op:<24} {cnt:>5}")
        summary_str = "\n".join(lines)
        logger.debug("summary():\n%s", summary_str)
        return summary_str


# ===========================================================================
# Pattern DSL
# ===========================================================================

class Pattern:
    """
    Lightweight pattern DSL for describing ONNX subgraph structures.

    Build patterns using arithmetic operators and helper constructors, then
    check whether a portion of a graph matches using the nested ``detect``
    class.

    Basic operators
    ---------------
    ``+``  → Add node
    ``*``  → Mul node
    ``**`` → Pow node
    calling a Pattern as a function → unary wrapper

    Wildcards
    ---------
    Use ``Pattern.any()`` to match any node regardless of op type.
    Use ``Pattern.const(v)`` to match a constant tensor whose value is *v*.
    Use ``Pattern.op("Sigmoid")`` as a named constructor.

    Example
    -------
    >>> x = Pattern.any()        # wildcard
    >>> c = Pattern.const(3.0)   # constant value
    >>> gelu = x * (Pattern.op("Erf")(x) + Pattern.const(1.0))
    >>> Pattern.detect(model, start_node).match(gelu)
    """

    def __init__(
        self,
        op: Optional[str] = None,
        inputs: Optional[List["Pattern"]] = None,
        value: Any = None,
    ) -> None:
        self.op: Optional[str] = op
        self.inputs: List["Pattern"] = inputs or []
        self.value: Any = value
        logger_pattern.debug(
            "Pattern created: op=%r, value=%r, #inputs=%d",
            self.op, self.value, len(self.inputs),
        )

    # ------------------------------------------------------------------
    # Named constructors
    # ------------------------------------------------------------------

    @classmethod
    def any(cls) -> "Pattern":
        """Wildcard: matches any node."""
        logger_pattern.debug("Pattern.any() created")
        return cls(op="*")

    @classmethod
    def const(cls, value: Any) -> "Pattern":
        """Match a constant / initializer tensor equal to *value*."""
        logger_pattern.debug("Pattern.const(%r) created", value)
        return cls(value=value)

    @classmethod
    def op(cls, op_type: str, *input_patterns: "Pattern") -> "Pattern":
        """
        Match a node with the given *op_type*, optionally constraining its
        parents via *input_patterns*.
        """
        logger_pattern.debug(
            "Pattern.op(%r) created with %d input pattern(s)", op_type, len(input_patterns)
        )
        return cls(op=op_type, inputs=list(input_patterns))

    # ------------------------------------------------------------------
    # Operator overloads (DSL sugar)
    # ------------------------------------------------------------------

    def __add__(self, other: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.__add__: constructing Add pattern")
        return Pattern("Add", [self, self._coerce(other)])

    def __radd__(self, other: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.__radd__: constructing Add pattern (reversed)")
        return Pattern("Add", [self._coerce(other), self])

    def __mul__(self, other: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.__mul__: constructing Mul pattern")
        return Pattern("Mul", [self, self._coerce(other)])

    def __rmul__(self, other: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.__rmul__: constructing Mul pattern (reversed)")
        return Pattern("Mul", [self._coerce(other), self])

    def __pow__(self, power: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.__pow__: constructing Pow pattern")
        return Pattern("Pow", [self, self._coerce(power)])

    def __sub__(self, other: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.__sub__: constructing Sub pattern")
        return Pattern("Sub", [self, self._coerce(other)])

    def __rsub__(self, other: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.__rsub__: constructing Sub pattern (reversed)")
        return Pattern("Sub", [self._coerce(other), self])

    def __truediv__(self, other: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.__truediv__: constructing Div pattern")
        return Pattern("Div", [self, self._coerce(other)])

    def __rtruediv__(self, other: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.__rtruediv__: constructing Div pattern (reversed)")
        return Pattern("Div", [self._coerce(other), self])

    def __neg__(self) -> "Pattern":
        logger_pattern.debug("Pattern.__neg__: constructing Neg pattern")
        return Pattern("Neg", [self])

    # ------------------------------------------------------------------
    # Activation function patterns
    # ------------------------------------------------------------------

    @classmethod
    def relu(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.relu() created")
        return cls.op("Relu")(x)

    @classmethod
    def sigmoid(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.sigmoid() created")
        return cls.op("Sigmoid")(x)

    @classmethod
    def tanh(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.tanh() created")
        return cls.op("Tanh")(x)

    @classmethod
    def leaky_relu(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.leaky_relu() created")
        return cls.op("LeakyRelu")(x)

    @classmethod
    def elu(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.elu() created")
        return cls.op("Elu")(x)

    @classmethod
    def selu(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.selu() created")
        return cls.op("Selu")(x)

    @classmethod
    def softplus(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.softplus() created")
        return cls.op("Softplus")(x)

    @classmethod
    def softsign(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.softsign() created")
        return cls.op("Softsign")(x)

    @classmethod
    def hardsigmoid(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.hardsigmoid() created")
        return cls.op("HardSigmoid")(x)

    @classmethod
    def hardswish(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.hardswish() created")
        return x * cls.hardsigmoid(x)

    @classmethod
    def silu(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.silu() created")
        return x * cls.sigmoid(x)

    @classmethod
    def swish(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.swish() created")
        return x * cls.sigmoid(x)

    @classmethod
    def gelu(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.gelu() created")
        return x * (cls.op("Erf")(x / cls.const(1.41421356237)) + cls.const(1.0)) * cls.const(0.5)

    @classmethod
    def gelu_tanh(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.gelu_tanh() created")
        return cls.const(0.5) * x * (
            cls.const(1.0)
            + cls.tanh(
                cls.const(0.7978845608) * (x + cls.const(0.044715) * (x ** cls.const(3.0)))
            )
        )

    @classmethod
    def mish(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.mish() created")
        return x * cls.tanh(cls.softplus(x))

    @classmethod
    def relu6(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.relu6() created")
        return Pattern("Clip", [cls._coerce(x), cls.const(0.0), cls.const(6.0)])

    @classmethod
    def softmax(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.softmax() created")
        return cls.op("Softmax")(x)

    @classmethod
    def log_softmax(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.log_softmax() created")
        return cls.op("LogSoftmax")(x)

    @classmethod
    def prelu(cls, x: "Pattern", slope: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.prelu() created")
        return cls.op("PRelu", x, slope)

    @classmethod
    def thresholded_relu(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.thresholded_relu() created")
        return cls.op("ThresholdedRelu")(x)

    def __call__(self, x: "Pattern") -> "Pattern":
        """Apply this pattern as a unary function: ``Pattern.op("Relu")(x)``."""
        logger_pattern.debug(
            "Pattern.__call__: applying %r as unary to input", self.op
        )
        if self.inputs:
            logger_pattern.error(
                "Pattern.__call__: pattern %r already has %d input(s); cannot call again",
                self.op, len(self.inputs),
            )
            raise ValueError(
                f"Pattern '{self.op}' already has {len(self.inputs)} input(s); "
                "cannot call again. Use Pattern.op(op_type, *inputs) for multi-input patterns."
            )
        return Pattern(self.op, [self._coerce(x)])

    # ------------------------------------------------------------------

    @staticmethod
    def _coerce(x: Any) -> "Pattern":
        if isinstance(x, Pattern):
            logger_pattern.debug("_coerce: value is already a Pattern (op=%r)", x.op)
            return x
        logger_pattern.debug("_coerce: wrapping raw value %r into Pattern.const", x)
        return Pattern(value=x)

    # ------------------------------------------------------------------
    # Nested detect class
    # ------------------------------------------------------------------

    class detect:
        """
        Match a :class:`Pattern` against a specific subgraph rooted at
        *start_node*.

        Parameters
        ----------
        model      : ModelProto or _GraphShim
                     The ONNX model (or a shim produced by ONNXQuery.matches).
        start_node : str or NodeProto
                     Root of the subgraph to check.
        end_node   : str or NodeProto, optional
                     Stop condition; if the traversal reaches this node the
                     branch is considered successful.

        Integration with ONNXQuery
        --------------------------
        For bulk matching across many nodes prefer
        ``ONNXQuery.matches(pattern)`` which calls this class internally.

        Example
        -------
        >>> detector = Pattern.detect(model, start_node="MatMul_0")
        >>> x = Pattern.any()
        >>> detector.match(x * Pattern.op("Sigmoid")(x))
        True
        """

        def __init__(
            self,
            model: Union[ModelProto, "_GraphShim"],
            start_node: Optional[Union[str, NodeProto]] = None,
            end_node: Optional[Union[str, NodeProto]] = None,
        ) -> None:
            # Accept either a real ModelProto or a lightweight shim
            if isinstance(model, _GraphShim):
                logger_detect.debug("detect: initialising from _GraphShim")
                self._nodes = model.nodes
                self._tensor_map = model.tensor_map
            else:
                logger_detect.debug("detect: initialising from ModelProto")
                self._nodes = list(model.graph.node)
                self._tensor_map = {
                    t.name: numpy_helper.to_array(t)
                    for t in model.graph.initializer
                }

            # Build output -> node map
            self._output_to_node: Dict[str, NodeProto] = {}
            for node in self._nodes:
                for out in node.output:
                    self._output_to_node[out] = node
            logger_detect.debug(
                "detect: output→node map built (%d entries)", len(self._output_to_node)
            )

            self.start = self._resolve(start_node)
            self.end   = self._resolve(end_node)
            logger_detect.debug(
                "detect: start=%r, end=%r",
                getattr(self.start, "name", None),
                getattr(self.end,   "name", None),
            )

        # --------------------------------------------------------------
        # Public API
        # --------------------------------------------------------------

        def match(self, pattern: "Pattern") -> bool:
            """Return True if the subgraph rooted at *start_node* matches *pattern*."""
            if self.start is None:
                logger_detect.debug("match(): start_node is None → False")
                return False
            logger_detect.debug(
                "match(): starting DFS from %r [%s]",
                self.start.name, self.start.op_type,
            )
            result = self._dfs(self.start, pattern, visited=frozenset())
            logger_detect.debug("match() → %s", result)
            return result

        def find_all(self, pattern: "Pattern") -> List[NodeProto]:
            """
            Scan nodes for subgraph matches against *pattern*.

            If *start_node* was provided at construction time, only nodes
            reachable as descendants of that node are scanned; otherwise the
            entire graph is searched.
            """
            if self.start is not None:
                logger_detect.debug(
                    "find_all(): scoping search to descendants of %r", self.start.name
                )
                # Collect descendants of start (inclusive) to scope the search
                descendants: Set[str] = set()
                frontier = [self.start]
                input_map: Dict[str, List[NodeProto]] = {}
                for node in self._nodes:
                    for inp in node.input:
                        input_map.setdefault(inp, []).append(node)
                while frontier:
                    current = frontier.pop()
                    if current.name in descendants:
                        continue
                    descendants.add(current.name)
                    for out in current.output:
                        for child in input_map.get(out, []):
                            if child.name not in descendants:
                                frontier.append(child)
                candidate_nodes = [n for n in self._nodes if n.name in descendants]
                logger_detect.debug(
                    "find_all(): %d candidate node(s) within descendant scope",
                    len(candidate_nodes),
                )
            else:
                logger_detect.debug(
                    "find_all(): no start_node, searching all %d nodes", len(self._nodes)
                )
                candidate_nodes = self._nodes

            shim = _GraphShim(self._nodes, self._tensor_map)
            matched = [
                node for node in candidate_nodes
                if node is not self.end
                and Pattern.detect(shim, start_node=node, end_node=self.end).match(pattern)
            ]
            logger_detect.debug("find_all() → %d match(es)", len(matched))
            return matched

        # --------------------------------------------------------------
        # Internal helpers
        # --------------------------------------------------------------

        def _resolve(
            self, node: Optional[Union[str, NodeProto]]
        ) -> Optional[NodeProto]:
            if node is None:
                logger_detect.debug("_resolve(None) → None")
                return None
            if hasattr(node, "op_type"):
                logger_detect.debug(
                    "_resolve: received NodeProto directly (%r [%s])",
                    node.name, node.op_type,
                )
                return node  # already NodeProto
            for n in self._nodes:
                if n.name == node:
                    logger_detect.debug(
                        "_resolve(%r): found node [%s]", node, n.op_type
                    )
                    return n
            logger_detect.error("_resolve(%r): node NOT found in graph", node)
            raise ValueError(f"Node not found: '{node}'")

        def _dfs(
            self,
            node: NodeProto,
            pattern: "Pattern",
            visited: frozenset,
        ) -> bool:
            logger_detect.debug(
                "_dfs: node=%r [%s], pattern.op=%r, pattern.value=%r, #pattern.inputs=%d, "
                "#visited=%d",
                node.name, node.op_type,
                pattern.op, pattern.value, len(pattern.inputs), len(visited),
            )

            # Stop condition
            if self.end is not None and node is self.end:
                logger_detect.debug(
                    "_dfs: reached end_node %r → True (stop condition)", node.name
                )
                return True

            # --- Constant / tensor leaf ---
            if pattern.value is not None:
                initializer_values = [
                    self._tensor_map[inp]
                    for inp in node.input
                    if inp and inp in self._tensor_map
                ]
                if not initializer_values:
                    # Also accept a bare Constant op node
                    if node.op_type == "Constant":
                        for attr in node.attribute:
                            if attr.name == "value":
                                arr = numpy_helper.to_array(attr.t)
                                match = bool(np.allclose(arr, pattern.value, atol=1e-3))
                                logger_detect.debug(
                                    "_dfs: Constant node %r, checking value≈%r → %s",
                                    node.name, pattern.value, match,
                                )
                                return match
                    logger_detect.debug(
                        "_dfs: pattern expects const %r but node %r [%s] has "
                        "no initializer inputs and is not a Constant op → False",
                        pattern.value, node.name, node.op_type,
                    )
                    return False
                matched_const = any(
                    np.allclose(arr, pattern.value, atol=1e-3)
                    for arr in initializer_values
                )
                logger_detect.debug(
                    "_dfs: node %r has %d initializer input(s); "
                    "const match for %r → %s",
                    node.name, len(initializer_values), pattern.value, matched_const,
                )
                return matched_const

            # --- Wildcard ---
            if pattern.op in ("*", "Input"):
                logger_detect.debug(
                    "_dfs: wildcard pattern %r matches node %r [%s] → True",
                    pattern.op, node.name, node.op_type,
                )
                return True

            # --- Op mismatch ---
            if pattern.op and node.op_type != pattern.op:
                logger_detect.debug(
                    "_dfs: op mismatch — expected %r, got %r for node %r → False",
                    pattern.op, node.op_type, node.name,
                )
                return False

            # No child constraints → op match is sufficient
            if not pattern.inputs:
                logger_detect.debug(
                    "_dfs: op %r matched node %r, no input constraints → True",
                    pattern.op, node.name,
                )
                return True

            # Collect parent nodes, separating initializer-backed edges for const patterns
            parents: List[NodeProto] = []
            for inp in node.input:
                if not inp:
                    continue
                parent = self._output_to_node.get(inp)
                if parent and parent.name not in visited:
                    parents.append(parent)
                    logger_detect.debug(
                        "_dfs: node %r input %r → parent %r [%s]",
                        node.name, inp, parent.name, parent.op_type,
                    )

            # For each pattern input that is a const, try to satisfy it directly from
            # the current node's initializer edges before requiring a parent node match.
            unmatched_patterns: List["Pattern"] = []
            used_initializers: Set[str] = set()

            for pat_idx, pat in enumerate(pattern.inputs):
                if pat.value is not None:
                    logger_detect.debug(
                        "_dfs: pattern input[%d] is const(%r); "
                        "checking initializers of node %r",
                        pat_idx, pat.value, node.name,
                    )
                    matched_init = False
                    for inp in node.input:
                        if inp and inp in self._tensor_map and inp not in used_initializers:
                            if np.allclose(self._tensor_map[inp], pat.value, atol=1e-3):
                                used_initializers.add(inp)
                                matched_init = True
                                logger_detect.debug(
                                    "_dfs:   initializer %r ≈ %r → const MATCHED",
                                    inp, pat.value,
                                )
                                break
                    if not matched_init:
                        logger_detect.debug(
                            "_dfs:   no initializer matched const %r → False", pat.value
                        )
                        return False
                else:
                    unmatched_patterns.append(pat)

            if len(parents) < len(unmatched_patterns):
                logger_detect.debug(
                    "_dfs: not enough parents (%d) for %d unmatched pattern(s) → False",
                    len(parents), len(unmatched_patterns),
                )
                return False

            # Commutative ops: try all permutations of non-const patterns vs parents
            if node.op_type in ("Add", "Mul"):
                logger_detect.debug(
                    "_dfs: node %r is commutative (%s); trying all permutations of "
                    "%d parent(s) against %d non-const pattern(s)",
                    node.name, node.op_type, len(parents), len(unmatched_patterns),
                )
                return self._match_commutative(
                    parents, unmatched_patterns, visited | frozenset([node.name])
                )

            # Ordered match for remaining (non-const) patterns
            logger_detect.debug(
                "_dfs: ordered match for %d non-const pattern(s) against %d parent(s)",
                len(unmatched_patterns), len(parents),
            )
            for pat_idx, (pat_child, parent) in enumerate(
                zip(unmatched_patterns, parents)
            ):
                logger_detect.debug(
                    "_dfs:   pattern input[%d] vs parent %r [%s]",
                    pat_idx, parent.name, parent.op_type,
                )
                if not self._dfs(parent, pat_child, visited | frozenset([node.name, parent.name])):

                    logger_detect.debug(
                        "_dfs:   ordered match failed at input[%d] → False", pat_idx
                    )
                    return False
            logger_detect.debug(
                "_dfs: all ordered inputs matched for node %r → True", node.name
            )
            return True

        def _match_commutative(
            self,
            parents: List[NodeProto],
            pattern_inputs: List["Pattern"],
            visited: frozenset,
        ) -> bool:
            n_perms = 1
            for i in range(len(parents), len(parents) - len(pattern_inputs), -1):
                n_perms *= i
            logger_detect.debug(
                "_match_commutative: testing up to %d permutation(s) "
                "of %d parent(s) vs %d pattern(s)",
                n_perms, len(parents), len(pattern_inputs),
            )
            for perm_idx, perm in enumerate(
                itertools.permutations(parents, len(pattern_inputs))
            ):
                logger_detect.debug(
                    "_match_commutative:   permutation[%d]: %s",
                    perm_idx, [p.name for p in perm],
                )
                if all(
                    self._dfs(p_node, pat, visited | frozenset([p_node.name]))
                    for p_node, pat in zip(perm, pattern_inputs)
                ):
                    logger_detect.debug(
                        "_match_commutative:   permutation[%d] succeeded → True", perm_idx
                    )
                    return True
            logger_detect.debug("_match_commutative: no permutation matched → False")
            return False


# ===========================================================================
# _GraphShim  (internal bridge object)
# ===========================================================================

class _GraphShim:
    """
    Minimal ModelProto-like object used to pass graph data from
    ONNXQuery into Pattern.detect without requiring a full ModelProto.
    """

    def __init__(self, nodes: List[NodeProto], tensor_map: TensorMap) -> None:
        self.nodes = nodes
        self.tensor_map = tensor_map
        logger_detect.debug(
            "_GraphShim created: %d nodes, %d tensors", len(nodes), len(tensor_map)
        )


# ===========================================================================
# Utility helpers
# ===========================================================================

def _attr_value(attr: Any) -> Any:
    """Extract a typed Python value from an ONNX AttributeProto."""
    import onnx
    t = attr.type
    if t == onnx.AttributeProto.FLOAT:
        return attr.f
    if t == onnx.AttributeProto.INT:
        return attr.i
    if t == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8")
    if t == onnx.AttributeProto.TENSOR:
        return numpy_helper.to_array(attr.t)
    if t == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    if t == onnx.AttributeProto.INTS:
        return list(attr.ints)
    if t == onnx.AttributeProto.STRINGS:
        return [s.decode("utf-8") for s in attr.strings]
    logger.debug("_attr_value: unrecognised attribute type %d for %r", t, attr.name)
    return None