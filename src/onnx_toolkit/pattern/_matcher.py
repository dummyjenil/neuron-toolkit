from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np
from onnx import numpy_helper
from onnx.onnx_pb import NodeProto

from onnx_toolkit._utils import _node_attrs
from onnx_toolkit.pattern.dsl import _ANY_OF, _CONST_PAT, _WILDCARD

if TYPE_CHECKING:
    from onnx_toolkit.pattern.detector import PatternDetector
    from onnx_toolkit.pattern.dsl import Pattern

log = logging.getLogger("onnx_toolkit.detect")


@dataclass
class MatchContext:
    """State management for a single pattern matching attempt."""

    detector: PatternDetector
    bindings: dict[str, NodeProto] = field(default_factory=dict)
    trail: list[NodeProto] = field(default_factory=list)
    # Maps id(Pattern) -> id(NodeProto) to enforce referential consistency
    # (e.g., if the same Pattern object appears twice in the DSL, it must match the same node)
    memo: dict[int, int] = field(default_factory=dict)
    visited: set[str] = field(default_factory=set)

    def snapshot(self) -> tuple[dict[str, NodeProto], list[NodeProto], dict[int, int], set[str]]:
        """Take a snapshot of the current state for backtracking."""
        return (dict(self.bindings), list(self.trail), dict(self.memo), set(self.visited))

    def restore(self, snap: tuple[dict[str, NodeProto], list[NodeProto], dict[int, int], set[str]]) -> None:
        """Restore state from a snapshot."""
        self.bindings, self.trail, self.memo, self.visited = snap


class MatchingMixin:
    """Mixin for matching logic in PatternDetector.

    This implementation uses a recursive depth-first search with backtracking and
    memoization to handle complex subgraph patterns, including wildcards,
    alternative paths, and commutative operators.
    """

    def _build_nx_graph(self: PatternDetector) -> nx.DiGraph:
        """Lazily build a NetworkX representation of the ONNX graph."""
        if self._nx_graph is None:
            g = nx.DiGraph()
            for n in self._nodes:
                g.add_node(n.name, proto=n)
                for inp in n.input:
                    parent = self._output_to_node.get(inp)
                    if parent:
                        g.add_edge(parent.name, n.name)
            self._nx_graph = g
        return self._nx_graph

    def _resolve(self: PatternDetector, node: str | NodeProto | None) -> NodeProto | None:
        """Resolve a node name or proto to a NodeProto instance."""
        if node is None or isinstance(node, NodeProto):
            return node
        for n in self._nodes:
            if n.name == node:
                return n
        msg = f"Node not found in graph: '{node}'"
        raise ValueError(msg)

    def _descendant_nodes(self: PatternDetector) -> list[NodeProto]:
        """Collect all descendants of self.start (inclusive)."""
        if self.start is None:
            msg = "start_node must be set to collect descendants"
            raise ValueError(msg)
        g = self._build_nx_graph()
        desc_names = nx.descendants(g, self.start.name) | {self.start.name}
        return [g.nodes[name]["proto"] for name in desc_names]

    def _match_recursive(
        self: PatternDetector,
        node: NodeProto,
        pattern: Pattern,
        ctx: MatchContext,
    ) -> bool:
        """Core recursive matching logic."""
        # 1. Referential Consistency Check
        pat_id = id(pattern)
        if pat_id in ctx.memo:
            return ctx.memo[pat_id] == id(node)

        # 2. Boundary Check
        if self.end is not None and node is self.end:
            return True

        # 3. Handle Special Pattern Types
        if pattern.op_type == _ANY_OF:
            return self._match_any_of(node, pattern, ctx)
        if pattern.op_type == _CONST_PAT:
            return self._match_const_pattern(node, pattern, ctx)
        if pattern.op_type == _WILDCARD:
            return self._match_wildcard(node, pattern, ctx)

        # 4. Op Type & Constraint Checks
        if pattern.op_type is not None and node.op_type != pattern.op_type:
            return False
        if not self._check_attrs(node, pattern) or not self._check_shape_dtype(node, pattern):
            return False

        # 5. Input Matching
        if not pattern.inputs:
            return self._finalize_match(node, pattern, ctx)

        parents = self._get_parent_nodes(node, ctx)
        if node.op_type in {"Add", "Mul"}:
            ok = self._match_commutative(node, parents, pattern, ctx)
        else:
            ok = self._match_ordered(node, parents, pattern, ctx)

        if ok:
            return self._finalize_match(node, pattern, ctx)
        return False

    def _match_any_of(self: PatternDetector, node: NodeProto, pattern: Pattern, ctx: MatchContext) -> bool:
        """Match if any of the alternatives match."""
        for alt in pattern._alternatives:
            snap = ctx.snapshot()
            if self._match_recursive(node, alt, ctx):
                if pattern._capture:
                    if pattern._capture in ctx.bindings and id(ctx.bindings[pattern._capture]) != id(node):
                        ctx.restore(snap)
                        continue
                    ctx.bindings[pattern._capture] = node
                ctx.memo[id(pattern)] = id(node)
                return True
            ctx.restore(snap)
        return False

    def _match_const_pattern(self: PatternDetector, node: NodeProto, pattern: Pattern, ctx: MatchContext) -> bool:
        """Match against a constant value."""
        if self._match_init_const_node(node, pattern.value):
            ctx.memo[id(pattern)] = id(node)
            return True
        return False

    def _match_wildcard(self: PatternDetector, node: NodeProto, pattern: Pattern, ctx: MatchContext) -> bool:
        """Match any node, subject to shape/dtype constraints."""
        if not self._check_shape_dtype(node, pattern):
            return False
        return self._finalize_match(node, pattern, ctx)

    def _match_ordered(
        self: PatternDetector,
        node: NodeProto,
        parents: list[NodeProto | None],
        pattern: Pattern,
        ctx: MatchContext,
    ) -> bool:
        """Match inputs in the exact order specified."""
        if len(parents) < len(pattern.inputs):
            return False

        snap = ctx.snapshot()
        ctx.visited.add(node.name)
        used_inits: set[str] = set()

        for i, pat in enumerate(pattern.inputs):
            if pat.op_type == _CONST_PAT:
                if not self._match_init_const_at_pos(node, i, pat.value, used_inits):
                    ctx.restore(snap)
                    return False
            else:
                parent = parents[i]
                if parent is None or not self._match_recursive(parent, pat, ctx):
                    ctx.restore(snap)
                    return False
        return True

    def _match_commutative(
        self: PatternDetector,
        node: NodeProto,
        parents: list[NodeProto | None],
        pattern: Pattern,
        ctx: MatchContext,
    ) -> bool:
        """Match inputs for commutative operations (Add, Mul)."""
        non_const_pats: list[Pattern] = []
        used_inits: set[str] = set()

        # First, match all constants
        for pat in pattern.inputs:
            if pat.op_type == _CONST_PAT:
                if not self._match_init_const_node(node, pat.value, used_inits):
                    return False
            else:
                non_const_pats.append(pat)

        # Match remaining patterns against actual node parents
        actual_parents = [p for p in parents if p is not None]
        if len(actual_parents) < len(non_const_pats):
            return False

        # Limit permutations to avoid explosion
        if len(actual_parents) > 8:
            log.warning("Too many parents for commutative match, limiting search")
            actual_parents = actual_parents[:8]

        for perm in itertools.permutations(actual_parents, len(non_const_pats)):
            snap = ctx.snapshot()
            ctx.visited.add(node.name)
            if all(self._match_recursive(p, pat, ctx) for p, pat in zip(perm, non_const_pats)):
                return True
            ctx.restore(snap)
        return False

    def _finalize_match(self: PatternDetector, node: NodeProto, pattern: Pattern, ctx: MatchContext) -> bool:
        """Record the successful match and update bindings/trail."""
        if not any(n is node for n in ctx.trail):
            ctx.trail.append(node)
        if pattern._capture:
            ctx.bindings[pattern._capture] = node
        ctx.memo[id(pattern)] = id(node)
        return True

    def _get_parent_nodes(self: PatternDetector, node: NodeProto, ctx: MatchContext) -> list[NodeProto | None]:
        """Get the parent nodes for a given node, filtering out visited ones."""
        parents: list[NodeProto | None] = []
        for inp in node.input:
            if not inp or inp in self._tensor_map:
                parents.append(None)
                continue
            parent = self._output_to_node.get(inp)
            if parent and parent.name not in ctx.visited:
                parents.append(parent)
            else:
                parents.append(None)
        return parents

    def _match_init_const_node(self: PatternDetector, node: NodeProto, value: object, used: set[str] | None = None) -> bool:
        """Match if any input to the node is an initializer/constant with the given value."""
        try:
            for i in range(len(node.input)):
                if self._match_init_const_at_pos(node, i, value, used or set()):
                    return True
            # Special case for Constant nodes themselves
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value":
                        val = numpy_helper.to_array(attr.t)
                        return bool(np.allclose(val, cast(Any, value), atol=1e-3))
        except (ValueError, TypeError):
            return False
        return False

    def _match_init_const_at_pos(
        self: PatternDetector, node: NodeProto, pos: int, value: object, used: set[str]
    ) -> bool:
        """Match a constant/initializer at a specific input position."""
        if pos >= len(node.input):
            return False
        inp = node.input[pos]
        if not inp or inp in used:
            return False

        try:
            # Case 1: Initializer
            if inp in self._tensor_map:
                if np.allclose(self._tensor_map[inp], cast(Any, value), atol=1e-3):
                    used.add(inp)
                    return True
            # Case 2: Constant node output
            parent = self._output_to_node.get(inp)
            if parent and parent.op_type == "Constant":
                for attr in parent.attribute:
                    if attr.name == "value":
                        val = numpy_helper.to_array(attr.t)
                        if np.allclose(val, cast(Any, value), atol=1e-3):
                            used.add(inp)
                            return True
        except (ValueError, TypeError):
            return False
        return False

    def _check_attrs(self: PatternDetector, node: NodeProto, pattern: Pattern) -> bool:
        """Check if node attributes satisfy pattern constraints."""
        if not pattern._constraints:
            return True
        attrs = _node_attrs(node)
        for attr_name, expected in pattern._constraints.items():
            if attr_name not in attrs:
                return False
            actual = attrs[attr_name]
            if callable(expected):
                if not cast("Callable[[Any], bool]", expected)(actual):
                    return False
            elif isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
                if not np.array_equal(actual, cast(Any, expected)):
                    return False
            elif actual != expected:
                return False
        return True

    def _check_shape_dtype(self: PatternDetector, node: NodeProto, pattern: Pattern) -> bool:
        """Check if node output shape and dtype satisfy pattern constraints."""
        if pattern._rank is None and pattern._dtype is None:
            return True
        if not node.output:
            return False
        rank, dtype = self._shape_info.get(node.output[0], (None, None))
        if pattern._rank is not None and rank != pattern._rank:
            return False
        if pattern._dtype is not None and dtype != pattern._dtype:
            return False
        return True
