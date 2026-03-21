"""
onnx_toolkit.pattern
=====================
Pattern DSL, MatchResult, and PatternDetector.

Pattern DSL
-----------
Build structural patterns using arithmetic operators and named constructors:

    x = Pattern.any().capture("x")
    p = Pattern.gelu(x)

    # Attribute constraint (exact value or callable predicate)
    depthwise = Pattern.op("Conv").where(group=lambda g: g > 1)

    # Union: match any of several alternatives
    act = Pattern.any_of(Pattern.relu(x), Pattern.gelu(x))

    # Shape / dtype constraints
    p = Pattern.any().with_output_rank(2).with_dtype("float16")

MatchResult
-----------
Returned by PatternDetector.match() on success:

    result.start    – NodeProto where the match began
    result.end      – NodeProto boundary (or start if no end_node given)
    result.nodes    – all NodeProtos visited during DFS (match subgraph)
    result.bindings – {capture_name: NodeProto} for every .capture() call

PatternDetector
---------------
    detector = PatternDetector(model_or_shim, start_node, end_node=None)
    result   = detector.match(pattern)       # → MatchResult | None
    results  = detector.find_all(pattern)    # → List[MatchResult]

End-node contract
-----------------
*end_node* is an **exclusive upstream boundary**.  When DFS would step into
a parent that IS end_node, that branch is immediately satisfied without
recursing into end_node.  The matched subgraph is {start_node … up-to but
not including end_node}.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Union

import numpy as np
from onnx import ModelProto, numpy_helper
from onnx.onnx_pb import NodeProto

from ._types import TensorMap
from ._utils import _GraphShim, _attr_value, _node_attrs, ShapeInfo

log_p = logging.getLogger("onnx_toolkit.pattern")
log_d = logging.getLogger("onnx_toolkit.detect")

# Sentinel op strings (never appear in real ONNX graphs)
_WILDCARD  = "__any__"
_ANY_OF    = "__any_of__"
_CONST_PAT = "__const__"


# ===========================================================================
# MatchResult
# ===========================================================================

@dataclass
class MatchResult:
    """
    Returned by PatternDetector.match() when a pattern matches.

    Attributes
    ----------
    start    : NodeProto          – root of the match (the start_node given to PatternDetector)
    end      : NodeProto          – terminal boundary node (end_node if provided, else start)
    nodes    : list[NodeProto]    – every node visited during the DFS, in visit order
    bindings : dict[str, NodeProto] – named captures populated by Pattern.capture()
    """
    start:    NodeProto
    end:      NodeProto
    nodes:    List[NodeProto]               = field(default_factory=list)
    bindings: Dict[str, NodeProto]          = field(default_factory=dict)

    def as_query(self) -> "ONNXQuery":  # type: ignore[name-defined]
        """
        Return the matched subgraph as an ONNXQuery for further analysis.

        Weight tensors are available only when this MatchResult was produced
        by ``ONNXQuery.match_results()`` (which attaches the parser's
        tensor_map automatically).  Results from a bare
        ``PatternDetector.match()`` call will have an empty tensor_map.
        """
        from .query import ONNXQuery  # local import — avoids circular dependency
        tm: TensorMap = getattr(self, "_tensor_map", {})
        return ONNXQuery(list(self.nodes), tm, list(self.nodes))


# ===========================================================================
# Pattern
# ===========================================================================

class Pattern:
    """
    Lightweight DSL for describing ONNX subgraph structures.

    Operators
    ---------
    +, -, *, /, **     → Add / Sub / Mul / Div / Pow nodes
    -x                 → Neg node
    pat(x)             → unary application: Pattern.op("Relu")(x)

    Constructors
    ------------
    Pattern.any()           wildcard — matches any single node
    Pattern.const(v)        matches a constant tensor ≈ v
    Pattern.op(op, *inputs) matches a specific op type
    Pattern.any_of(*pats)   union — succeeds if any alternative matches

    Modifiers (return a new Pattern — do not mutate)
    ---------
    .capture(name)          bind matched node to *name* in MatchResult.bindings
    .where(**constraints)   attribute constraints; values may be plain values
                            or single-argument callables acting as predicates
    .with_output_rank(r)    shape constraint: first output must have rank *r*
    .with_dtype(dt)         dtype constraint: first output must have dtype *dt*
                            (numpy dtype string, e.g. "float32")

    Activation helpers
    ------------------
    Pattern.relu, sigmoid, tanh, leaky_relu, elu, selu, softplus, softsign,
    hardsigmoid, hardswish, silu / swish, gelu, gelu_tanh, mish, relu6,
    softmax, log_softmax, prelu, thresholded_relu
    """

    def __init__(
        self,
        op:           Optional[str]       = None,
        inputs:       Optional[List["Pattern"]] = None,
        value:        Any                 = None,
        *,
        _alternatives: Optional[List["Pattern"]] = None,
        _capture:     Optional[str]       = None,
        _constraints: Optional[Dict[str, Any]] = None,
        _rank:        Optional[int]       = None,
        _dtype:       Optional[str]       = None,
    ) -> None:
        self.op            = op
        self.inputs        = inputs or []
        self.value         = value
        self._alternatives = _alternatives or []
        self._capture      = _capture
        self._constraints  = _constraints or {}
        self._rank         = _rank
        self._dtype        = _dtype

    # ------------------------------------------------------------------
    # Named constructors
    # ------------------------------------------------------------------

    @classmethod
    def any(cls) -> "Pattern":
        """Wildcard: matches any single node."""
        return cls(op=_WILDCARD)

    @classmethod
    def const(cls, value: Any) -> "Pattern":
        """Match a constant / initializer tensor ≈ *value*."""
        return cls(op=_CONST_PAT, value=value)

    @classmethod
    def op(cls, op_type: str, *input_patterns: "Pattern") -> "Pattern":
        """Match a node with the given *op_type*, constraining its inputs."""
        return cls(op=op_type, inputs=list(input_patterns))

    @classmethod
    def any_of(cls, *alternatives: "Pattern") -> "Pattern":
        """
        Union pattern: succeeds if **any** of *alternatives* matches.
        The first successful alternative wins; its bindings are kept.
        """
        if len(alternatives) < 2:
            raise ValueError("any_of() requires at least 2 alternatives")
        return cls(op=_ANY_OF, _alternatives=list(alternatives))

    # ------------------------------------------------------------------
    # Modifiers
    # ------------------------------------------------------------------

    def capture(self, name: str) -> "Pattern":
        """Bind the matched node to *name* in MatchResult.bindings."""
        return Pattern(
            self.op, list(self.inputs), self.value,
            _alternatives=list(self._alternatives),
            _capture=name,
            _constraints=dict(self._constraints),
            _rank=self._rank, _dtype=self._dtype,
        )

    def where(self, **constraints: Any) -> "Pattern":
        """
        Add attribute constraints.  Each keyword is an attribute name;
        the value is either a literal (checked with ==) or a callable
        predicate ``fn(attr_value) -> bool``.

        Example::

            Pattern.op("Conv").where(group=lambda g: g > 1)
            Pattern.op("Gather").where(axis=0)
        """
        merged = {**self._constraints, **constraints}
        return Pattern(
            self.op, list(self.inputs), self.value,
            _alternatives=list(self._alternatives),
            _capture=self._capture,
            _constraints=merged,
            _rank=self._rank, _dtype=self._dtype,
        )

    def with_output_rank(self, rank: int) -> "Pattern":
        """Require the node's first output to have the given tensor rank."""
        return Pattern(
            self.op, list(self.inputs), self.value,
            _alternatives=list(self._alternatives),
            _capture=self._capture,
            _constraints=dict(self._constraints),
            _rank=rank, _dtype=self._dtype,
        )

    def with_dtype(self, dtype: str) -> "Pattern":
        """
        Require the node's first output to have *dtype* (numpy string,
        e.g. ``"float32"``, ``"float16"``).
        """
        return Pattern(
            self.op, list(self.inputs), self.value,
            _alternatives=list(self._alternatives),
            _capture=self._capture,
            _constraints=dict(self._constraints),
            _rank=self._rank, _dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Arithmetic DSL
    # ------------------------------------------------------------------

    def __add__(self, other: Any) -> "Pattern":
        return Pattern("Add", [self, _coerce(other)])
    def __radd__(self, other: Any) -> "Pattern":
        return Pattern("Add", [_coerce(other), self])
    def __mul__(self, other: Any) -> "Pattern":
        return Pattern("Mul", [self, _coerce(other)])
    def __rmul__(self, other: Any) -> "Pattern":
        return Pattern("Mul", [_coerce(other), self])
    def __pow__(self, power: Any) -> "Pattern":
        return Pattern("Pow", [self, _coerce(power)])
    def __sub__(self, other: Any) -> "Pattern":
        return Pattern("Sub", [self, _coerce(other)])
    def __rsub__(self, other: Any) -> "Pattern":
        return Pattern("Sub", [_coerce(other), self])
    def __truediv__(self, other: Any) -> "Pattern":
        return Pattern("Div", [self, _coerce(other)])
    def __rtruediv__(self, other: Any) -> "Pattern":
        return Pattern("Div", [_coerce(other), self])
    def __neg__(self) -> "Pattern":
        return Pattern("Neg", [self])

    def __call__(self, x: Any) -> "Pattern":
        """Unary application: ``Pattern.op("Relu")(x)``."""
        if self.inputs:
            raise ValueError(
                f"Pattern '{self.op}' already has inputs; "
                "use Pattern.op(op_type, *inputs) for multi-input patterns."
            )
        return Pattern(
            self.op, [_coerce(x)], self.value,
            _alternatives=list(self._alternatives),
            _capture=self._capture,
            _constraints=dict(self._constraints),
            _rank=self._rank, _dtype=self._dtype,
        )

    # ------------------------------------------------------------------
    # Activation helpers
    # ------------------------------------------------------------------

    @classmethod
    def relu(cls, x: "Pattern") -> "Pattern":
        return cls.op("Relu")(x)
    @classmethod
    def sigmoid(cls, x: "Pattern") -> "Pattern":
        return cls.op("Sigmoid")(x)
    @classmethod
    def tanh(cls, x: "Pattern") -> "Pattern":
        return cls.op("Tanh")(x)
    @classmethod
    def leaky_relu(cls, x: "Pattern") -> "Pattern":
        return cls.op("LeakyRelu")(x)
    @classmethod
    def elu(cls, x: "Pattern") -> "Pattern":
        return cls.op("Elu")(x)
    @classmethod
    def selu(cls, x: "Pattern") -> "Pattern":
        return cls.op("Selu")(x)
    @classmethod
    def softplus(cls, x: "Pattern") -> "Pattern":
        return cls.op("Softplus")(x)
    @classmethod
    def softsign(cls, x: "Pattern") -> "Pattern":
        return cls.op("Softsign")(x)
    @classmethod
    def hardsigmoid(cls, x: "Pattern") -> "Pattern":
        return cls.op("HardSigmoid")(x)
    @classmethod
    def hardswish(cls, x: "Pattern") -> "Pattern":
        return x * cls.hardsigmoid(x)
    @classmethod
    def silu(cls, x: "Pattern") -> "Pattern":
        return x * cls.sigmoid(x)
    @classmethod
    def swish(cls, x: "Pattern") -> "Pattern":
        return x * cls.sigmoid(x)
    @classmethod
    def gelu(cls, x: "Pattern") -> "Pattern":
        return x * (cls.op("Erf")(x / cls.const(1.41421356237)) + cls.const(1.0)) * cls.const(0.5)
    @classmethod
    def gelu_tanh(cls, x: "Pattern") -> "Pattern":
        return cls.const(0.5) * x * (
            cls.const(1.0) + cls.tanh(
                cls.const(0.7978845608) * (x + cls.const(0.044715) * (x ** cls.const(3.0)))
            )
        )
    @classmethod
    def mish(cls, x: "Pattern") -> "Pattern":
        return x * cls.tanh(cls.softplus(x))
    @classmethod
    def relu6(cls, x: "Pattern") -> "Pattern":
        return Pattern("Clip", [_coerce(x), cls.const(0.0), cls.const(6.0)])
    @classmethod
    def softmax(cls, x: "Pattern") -> "Pattern":
        return cls.op("Softmax")(x)
    @classmethod
    def log_softmax(cls, x: "Pattern") -> "Pattern":
        return cls.op("LogSoftmax")(x)
    @classmethod
    def prelu(cls, x: "Pattern", slope: "Pattern") -> "Pattern":
        return cls.op("PRelu", x, slope)
    @classmethod
    def thresholded_relu(cls, x: "Pattern") -> "Pattern":
        return cls.op("ThresholdedRelu")(x)


def _coerce(x: Any) -> Pattern:
    """Wrap a raw value in Pattern.const if it isn't already a Pattern."""
    return x if isinstance(x, Pattern) else Pattern(op=_CONST_PAT, value=x)


# ===========================================================================
# PatternDetector
# ===========================================================================

class PatternDetector:
    """
    Match a :class:`Pattern` against a subgraph of an ONNX model.

    Parameters
    ----------
    model      : ModelProto or _GraphShim
    start_node : str | NodeProto  – root of the region to match (inclusive)
    end_node   : str | NodeProto  – exclusive upstream boundary (optional)

    End-node contract
    -----------------
    When DFS would step into a parent that IS end_node, the branch is
    immediately satisfied.  end_node is never part of the matched subgraph.

        [end_node] → [A] → [B] → [start_node]
        matched subgraph = {start_node, B, A}

    match() → MatchResult | None
    find_all() → List[MatchResult]
    """

    def __init__(
        self,
        model:      Union[ModelProto, _GraphShim],
        start_node: Optional[Union[str, NodeProto]] = None,
        end_node:   Optional[Union[str, NodeProto]] = None,
    ) -> None:
        if isinstance(model, _GraphShim):
            self._nodes      = model.nodes
            self._tensor_map = model.tensor_map
            self._shape_info: ShapeInfo = model.shape_info
        else:
            self._nodes      = list(model.graph.node)
            self._tensor_map = {
                t.name: numpy_helper.to_array(t)
                for t in model.graph.initializer
            }
            self._shape_info = {}

        self._output_to_node: Dict[str, NodeProto] = {
            out: n for n in self._nodes for out in n.output
        }
        self.start = self._resolve(start_node)
        self.end   = self._resolve(end_node)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, pattern: Pattern) -> Optional[MatchResult]:
        """
        Match *pattern* against the subgraph rooted at start_node.

        Returns a :class:`MatchResult` on success, ``None`` on failure.
        The result's `.end` is end_node if it was reached, otherwise start_node.
        """
        if self.start is None:
            return None
        bindings: Dict[str, NodeProto] = {}
        visited:  List[NodeProto]      = []
        ok = self._dfs(self.start, pattern, frozenset(), bindings, visited)
        if not ok:
            return None
        terminal = self.end if self.end is not None else self.start
        return MatchResult(
            start    = self.start,
            end      = terminal,
            nodes    = visited,
            bindings = bindings,
        )

    def find_all(self, pattern: Pattern) -> List[MatchResult]:
        """
        Scan all (or descendant) nodes for pattern matches.

        Returns one :class:`MatchResult` per matching start node.
        Shares the output→node map across all candidates for efficiency.
        """
        candidates = self._descendant_nodes() if self.start is not None else self._nodes
        shim = _GraphShim(self._nodes, self._tensor_map, self._shape_info)
        results: List[MatchResult] = []
        for node in candidates:
            if node is self.end:
                continue
            det = PatternDetector(shim, start_node=node, end_node=self.end)
            det._output_to_node = self._output_to_node  # share — no rebuild
            r = det.match(pattern)
            if r is not None:
                results.append(r)
        log_d.debug("find_all() → %d match(es)", len(results))
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve(self, node: Optional[Union[str, NodeProto]]) -> Optional[NodeProto]:
        if node is None:
            return None
        if hasattr(node, "op_type"):
            return node  # type: ignore[return-value]
        for n in self._nodes:
            if n.name == node:
                return n
        raise ValueError(f"Node not found in graph: '{node}'")

    def _descendant_nodes(self) -> List[NodeProto]:
        """Collect all descendants of self.start (inclusive) via BFS."""
        assert self.start is not None
        input_map: Dict[str, List[NodeProto]] = {}
        for n in self._nodes:
            for inp in n.input:
                input_map.setdefault(inp, []).append(n)
        visited: Set[str] = set()
        frontier = [self.start]
        result: List[NodeProto] = []
        while frontier:
            cur = frontier.pop()
            if cur.name in visited:
                continue
            visited.add(cur.name)
            result.append(cur)
            for out in cur.output:
                for child in input_map.get(out, []):
                    if child.name not in visited:
                        frontier.append(child)
        return result

    # ------------------------------------------------------------------
    # DFS core
    # ------------------------------------------------------------------

    def _dfs(
        self,
        node:     NodeProto,
        pattern:  Pattern,
        visited:  FrozenSet[str],
        bindings: Dict[str, NodeProto],
        trail:    List[NodeProto],
    ) -> bool:
        """
        Recursively match *pattern* against *node* and its upstream parents.

        *bindings* and *trail* are mutated in-place by successful branches
        and restored on backtrack.
        """
        log_d.debug("_dfs: node=%r [%s] pattern.op=%r", node.name, node.op_type, pattern.op)

        # ---- End-node boundary ------------------------------------------
        if self.end is not None and node is self.end:
            log_d.debug("_dfs: boundary end_node %r → True", node.name)
            return True

        # ---- Union pattern ----------------------------------------------
        if pattern.op == _ANY_OF:
            for alt in pattern._alternatives:
                # Snapshot bindings before trying each alternative
                snap_b = dict(bindings)
                snap_t = len(trail)
                if self._dfs(node, alt, visited, bindings, trail):
                    # Apply capture for the union node itself, if any
                    if pattern._capture:
                        bindings[pattern._capture] = node
                    return True
                # Restore on failure
                bindings.clear()
                bindings.update(snap_b)
                del trail[snap_t:]
            return False

        # ---- Constant pattern -------------------------------------------
        if pattern.op == _CONST_PAT:
            return self._match_const(node, pattern.value)

        # ---- Wildcard ---------------------------------------------------
        if pattern.op == _WILDCARD:
            if not self._check_shape_dtype(node, pattern):
                return False
            self._record(node, pattern, bindings, trail)
            return True

        # ---- Op-type check ----------------------------------------------
        if pattern.op is not None and node.op_type != pattern.op:
            return False

        # ---- Attribute constraints --------------------------------------
        if not self._check_attrs(node, pattern):
            return False

        # ---- Shape / dtype constraints ----------------------------------
        if not self._check_shape_dtype(node, pattern):
            return False

        # ---- No input constraints → match complete ----------------------
        if not pattern.inputs:
            self._record(node, pattern, bindings, trail)
            return True

        # ---- Gather parents (non-initializer edges) ---------------------
        parents = self._parents(node, visited)

        # Separate const-pattern inputs (satisfied by initializer edges)
        # from structural patterns (require parent nodes).
        non_const_pats: List[Pattern] = []
        used_inits: Set[str] = set()

        for pat in pattern.inputs:
            if pat.op == _CONST_PAT:
                if not self._match_init_const(node, pat.value, used_inits):
                    return False
            else:
                non_const_pats.append(pat)

        if len(parents) < len(non_const_pats):
            return False

        new_visited = visited | {node.name}

        # ---- Commutative ops: try all permutations ----------------------
        if node.op_type in ("Add", "Mul"):
            ok = self._try_commutative(
                parents, non_const_pats, new_visited, bindings, trail
            )
        else:
            ok = self._try_ordered(
                parents, non_const_pats, new_visited, bindings, trail
            )

        if ok:
            self._record(node, pattern, bindings, trail)
        return ok

    # ------------------------------------------------------------------
    # Match helpers
    # ------------------------------------------------------------------

    def _match_const(self, node: NodeProto, value: Any) -> bool:
        """Check whether *node* is or carries a constant ≈ *value*."""
        inits = [
            self._tensor_map[inp]
            for inp in node.input
            if inp and inp in self._tensor_map
        ]
        if inits:
            return any(np.allclose(a, value, atol=1e-3) for a in inits)
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    return bool(np.allclose(numpy_helper.to_array(attr.t), value, atol=1e-3))
        return False

    def _match_init_const(
        self, node: NodeProto, value: Any, used: Set[str]
    ) -> bool:
        """Satisfy one const-pattern input from node's initializer edges."""
        for inp in node.input:
            if inp and inp in self._tensor_map and inp not in used:
                if np.allclose(self._tensor_map[inp], value, atol=1e-3):
                    used.add(inp)
                    return True
        return False

    def _check_attrs(self, node: NodeProto, pattern: Pattern) -> bool:
        """Verify all attribute constraints stored in *pattern*."""
        if not pattern._constraints:
            return True
        attrs = _node_attrs(node)
        for attr_name, expected in pattern._constraints.items():
            if attr_name not in attrs:
                return False
            actual = attrs[attr_name]
            if callable(expected):
                if not expected(actual):
                    return False
            else:
                if actual != expected:
                    return False
        return True

    def _check_shape_dtype(self, node: NodeProto, pattern: Pattern) -> bool:
        """Verify rank / dtype constraints against the node's first output."""
        if pattern._rank is None and pattern._dtype is None:
            return True
        if not node.output:
            return False
        first_out = node.output[0]
        rank, dtype = self._shape_info.get(first_out, (None, None))
        if pattern._rank is not None and rank != pattern._rank:
            return False
        if pattern._dtype is not None and dtype != pattern._dtype:
            return False
        return True

    def _parents(
        self, node: NodeProto, visited: FrozenSet[str]
    ) -> List[NodeProto]:
        """Collect unvisited parent nodes for non-initializer inputs."""
        parents: List[NodeProto] = []
        for inp in node.input:
            if not inp or inp in self._tensor_map:
                continue
            parent = self._output_to_node.get(inp)
            if parent and parent.name not in visited:
                parents.append(parent)
        return parents

    def _try_ordered(
        self,
        parents:   List[NodeProto],
        patterns:  List[Pattern],
        visited:   FrozenSet[str],
        bindings:  Dict[str, NodeProto],
        trail:     List[NodeProto],
    ) -> bool:
        for parent, pat in zip(parents, patterns):
            snap_b = dict(bindings)
            snap_t = len(trail)
            if not self._dfs(parent, pat, visited | {parent.name}, bindings, trail):
                bindings.clear(); bindings.update(snap_b)
                del trail[snap_t:]
                return False
        return True

    def _try_commutative(
        self,
        parents:   List[NodeProto],
        patterns:  List[Pattern],
        visited:   FrozenSet[str],
        bindings:  Dict[str, NodeProto],
        trail:     List[NodeProto],
    ) -> bool:
        for perm in itertools.permutations(parents, len(patterns)):
            snap_b = dict(bindings)
            snap_t = len(trail)
            if all(
                self._dfs(p, pat, visited | {p.name}, bindings, trail)
                for p, pat in zip(perm, patterns)
            ):
                return True
            bindings.clear(); bindings.update(snap_b)
            del trail[snap_t:]
        return False

    @staticmethod
    def _record(
        node:     NodeProto,
        pattern:  Pattern,
        bindings: Dict[str, NodeProto],
        trail:    List[NodeProto],
    ) -> None:
        """Record a successful node visit (deduplicating the trail)."""
        if not any(n is node for n in trail):
            trail.append(node)
        if pattern._capture:
            bindings[pattern._capture] = node
