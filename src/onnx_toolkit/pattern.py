"""
onnx_toolkit.pattern
=====================
Pattern DSL and PatternDetector for describing and matching ONNX subgraph
structures.

Pattern
-------
Build patterns with arithmetic operators and named constructors, then hand
them to PatternDetector.match().

    x    = Pattern.any()
    gelu = x * (Pattern.op("Erf")(x / Pattern.const(1.41421356237))
                + Pattern.const(1.0)) * Pattern.const(0.5)

PatternDetector
---------------
Replaces the previous nested ``Pattern.detect`` class.  Constructed from a
model (or _GraphShim) plus a mandatory *start_node* and an optional
*end_node*.

End-node contract
~~~~~~~~~~~~~~~~~
The *end_node* is the **exclusive boundary** of the subgraph: it is the first
node the DFS is *not* allowed to enter.  Whenever the traversal would step
into a parent that is exactly *end_node*, that branch is treated as
successfully terminated — the pattern input at that position is considered
satisfied without further recursion.

This makes the semantics concrete and graph-theoretically sound:

  • The matched subgraph consists of all nodes visited between *start_node*
    (inclusive) and *end_node* (exclusive).
  • Callers can supply *end_node* to prevent the match from "leaking" into
    upstream subgraphs they don't care about.

match() return value
~~~~~~~~~~~~~~~~~~~~
``match()`` previously returned ``bool``.  It now returns the **end node**
(``NodeProto``) when a match is found, or ``None`` on failure.

  • If no *end_node* was provided the successful match terminates at a leaf
    (graph input / initializer), and ``match()`` returns the *start_node*
    itself so the caller still gets a concrete node back.
  • If an *end_node* was provided and the DFS reaches it, ``match()``
    returns that ``NodeProto``.

Logging
-------
  onnx_toolkit.pattern  – Pattern construction
  onnx_toolkit.detect   – PatternDetector DFS walk & decisions
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, FrozenSet, List, Optional, Set, Union

import numpy as np
from onnx import ModelProto, numpy_helper
from onnx.onnx_pb import NodeProto

from ._types import TensorMap
from ._utils import _GraphShim

logger_pattern = logging.getLogger("onnx_toolkit.pattern")
logger_detect  = logging.getLogger("onnx_toolkit.detect")


# ===========================================================================
# Pattern
# ===========================================================================

class Pattern:
    """
    Lightweight DSL for describing ONNX subgraph structures.

    Basic operators
    ---------------
    ``+``   → Add node
    ``*``   → Mul node
    ``**``  → Pow node
    ``-``   → Sub node  (or unary Neg)
    ``/``   → Div node
    calling a Pattern → unary wrapper

    Wildcards
    ---------
    ``Pattern.any()``      – match any node regardless of op type.
    ``Pattern.const(v)``   – match a constant tensor whose value ≈ v.
    ``Pattern.op("Relu")`` – match a specific op type.
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
            "Pattern.op(%r) created with %d input pattern(s)",
            op_type, len(input_patterns),
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
        return (
            x
            * (cls.op("Erf")(x / cls.const(1.41421356237)) + cls.const(1.0))
            * cls.const(0.5)
        )

    @classmethod
    def gelu_tanh(cls, x: "Pattern") -> "Pattern":
        logger_pattern.debug("Pattern.gelu_tanh() created")
        return cls.const(0.5) * x * (
            cls.const(1.0)
            + cls.tanh(
                cls.const(0.7978845608)
                * (x + cls.const(0.044715) * (x ** cls.const(3.0)))
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

    # ------------------------------------------------------------------

    @staticmethod
    def _coerce(x: Any) -> "Pattern":
        if isinstance(x, Pattern):
            logger_pattern.debug("_coerce: value is already a Pattern (op=%r)", x.op)
            return x
        logger_pattern.debug("_coerce: wrapping raw value %r into Pattern.const", x)
        return Pattern(value=x)


# ===========================================================================
# PatternDetector
# ===========================================================================

class PatternDetector:
    """
    Match a :class:`Pattern` against a specific subgraph rooted at
    *start_node*.

    Parameters
    ----------
    model      : ModelProto or _GraphShim
                 The ONNX model (or a shim produced by ONNXQuery.matches).
    start_node : str or NodeProto
                 Root of the subgraph to match (inclusive).
    end_node   : str or NodeProto, optional
                 Exclusive boundary of the subgraph.  When the DFS traversal
                 would step into a parent that is exactly *end_node*, that
                 branch is treated as successfully terminated — the pattern
                 input at that position is satisfied without further
                 recursion.  This prevents the match from "leaking" into
                 upstream subgraphs the caller does not want to include.

    End-node contract
    -----------------
    *end_node* is **exclusive**: it is never itself part of the matched
    subgraph.  Think of it as the first node *upstream* of the region you
    care about.  For example::

        [end_node] → [A] → [B] → [start_node]

    Calling ``match()`` with this setup will match the subgraph
    {start_node, B, A} and stop before entering end_node.

    match() return value
    --------------------
    Returns the **terminating** ``NodeProto`` on success, or ``None`` on
    failure:

    * If *end_node* was given and the DFS reached it: returns *end_node*.
    * If no *end_node* was given (match terminates at a graph leaf):
      returns *start_node* so the caller always gets a concrete node.

    Integration with ONNXQuery
    --------------------------
    Prefer ``ONNXQuery.matches(pattern)`` for bulk matching; it calls this
    class internally and shares the output→node map across all candidates.

    Example
    -------
    >>> x        = Pattern.any()
    >>> detector = PatternDetector(model, start_node="MatMul_0")
    >>> end      = detector.match(x * Pattern.op("Sigmoid")(x))
    >>> if end:
    ...     print("Pattern ends at:", end.name)
    """

    def __init__(
        self,
        model: Union[ModelProto, _GraphShim],
        start_node: Optional[Union[str, NodeProto]] = None,
        end_node: Optional[Union[str, NodeProto]] = None,
    ) -> None:
        if isinstance(model, _GraphShim):
            logger_detect.debug("PatternDetector: initialising from _GraphShim")
            self._nodes = model.nodes
            self._tensor_map = model.tensor_map
        else:
            logger_detect.debug("PatternDetector: initialising from ModelProto")
            self._nodes = list(model.graph.node)
            self._tensor_map: TensorMap = {
                t.name: numpy_helper.to_array(t)
                for t in model.graph.initializer
            }

        # Build output → node map
        self._output_to_node: Dict[str, NodeProto] = {}
        for node in self._nodes:
            for out in node.output:
                self._output_to_node[out] = node
        logger_detect.debug(
            "PatternDetector: output→node map built (%d entries)",
            len(self._output_to_node),
        )

        self.start: Optional[NodeProto] = self._resolve(start_node)
        self.end:   Optional[NodeProto] = self._resolve(end_node)
        logger_detect.debug(
            "PatternDetector: start=%r, end=%r",
            getattr(self.start, "name", None),
            getattr(self.end,   "name", None),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, pattern: Pattern) -> Optional[NodeProto]:
        """
        Try to match *pattern* against the subgraph rooted at *start_node*.

        Returns
        -------
        NodeProto
            The terminating node of the match:

            * *end_node* if it was provided and reached during DFS.
            * *start_node* if no *end_node* was provided (match terminated
              at a graph leaf).

        None
            The pattern did not match.
        """
        if self.start is None:
            logger_detect.debug("match(): start_node is None → None")
            return None
        logger_detect.debug(
            "match(): starting DFS from %r [%s]",
            self.start.name, self.start.op_type,
        )
        result = self._dfs(self.start, pattern, visited=frozenset())
        if result:
            # Return end_node when reached, otherwise start_node as the
            # concrete anchor of a leaf-terminated match.
            terminal = self.end if self.end is not None else self.start
            logger_detect.debug(
                "match() → MATCHED, terminal node=%r [%s]",
                terminal.name, terminal.op_type,
            )
            return terminal
        logger_detect.debug("match() → None (no match)")
        return None

    def find_all(self, pattern: Pattern) -> List[NodeProto]:
        """
        Scan nodes for subgraph matches against *pattern*.

        If *start_node* was provided at construction time, only nodes
        reachable as descendants of that node are scanned; otherwise the
        entire graph is searched.

        Returns a list of *start* nodes of each matching subgraph.
        """
        if self.start is not None:
            logger_detect.debug(
                "find_all(): scoping search to descendants of %r", self.start.name
            )
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
        matched: List[NodeProto] = []
        for node in candidate_nodes:
            if node is self.end:
                continue
            detector = PatternDetector(shim, start_node=node, end_node=self.end)
            detector._output_to_node = self._output_to_node  # share map
            if detector.match(pattern) is not None:
                matched.append(node)

        logger_detect.debug("find_all() → %d match(es)", len(matched))
        return matched

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
            return node
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
        pattern: Pattern,
        visited: FrozenSet[str],
    ) -> bool:
        logger_detect.debug(
            "_dfs: node=%r [%s], pattern.op=%r, pattern.value=%r, "
            "#pattern.inputs=%d, #visited=%d",
            node.name, node.op_type,
            pattern.op, pattern.value, len(pattern.inputs), len(visited),
        )

        # ------------------------------------------------------------------
        # End-node boundary check
        #
        # When the DFS arrives AT a node that is the declared end_node, it
        # means the traversal has reached the exclusive boundary of the
        # subgraph.  We treat this as a successful termination: the pattern
        # input that pointed here is considered satisfied without inspecting
        # end_node's own structure.  This is the correct graph-theoretic
        # interpretation: end_node is the first node *outside* the region
        # being matched.
        # ------------------------------------------------------------------
        if self.end is not None and node is self.end:
            logger_detect.debug(
                "_dfs: reached end_node %r — boundary satisfied → True",
                node.name,
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
                "_dfs: wildcard %r matches node %r [%s] → True",
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

        # Collect parent nodes
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

        # Satisfy const-pattern inputs directly from initializer edges
        unmatched_patterns: List[Pattern] = []
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
                    if (
                        inp
                        and inp in self._tensor_map
                        and inp not in used_initializers
                    ):
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

        new_visited = visited | frozenset([node.name])

        # Commutative ops: try all permutations
        if node.op_type in ("Add", "Mul"):
            logger_detect.debug(
                "_dfs: commutative op %s on node %r; trying permutations "
                "of %d parent(s) against %d non-const pattern(s)",
                node.op_type, node.name, len(parents), len(unmatched_patterns),
            )
            return self._match_commutative(parents, unmatched_patterns, new_visited)

        # Ordered match
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
            if not self._dfs(parent, pat_child, new_visited | frozenset([parent.name])):
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
        pattern_inputs: List[Pattern],
        visited: FrozenSet[str],
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
