from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np

from neuron_toolkit.pattern.dsl import _ANY_OF, _CONST_PAT, _WILDCARD

if TYPE_CHECKING:
    from neuron_toolkit.pattern.detector import PatternDetector
    from neuron_toolkit.pattern.dsl import Pattern

log = logging.getLogger("neuron_toolkit.detect")


@dataclass
class MatchContext:
    """State management for a single pattern matching attempt."""

    detector: PatternDetector
    bindings: dict[str, Any] = field(default_factory=dict)
    trail: list[Any] = field(default_factory=list)
    # Maps id(Pattern) -> id(Node) to enforce referential consistency
    memo: dict[int, int] = field(default_factory=dict)
    visited: set[str] = field(default_factory=set)

    def snapshot(self) -> tuple[dict[str, Any], list[Any], dict[int, int], set[str]]:
        """Take a snapshot of the current state for backtracking."""
        return (
            dict(self.bindings),
            list(self.trail),
            dict(self.memo),
            set(self.visited),
        )

    def restore(
        self, snap: tuple[dict[str, Any], list[Any], dict[int, int], set[str]]
    ) -> None:
        """Restore state from a snapshot."""
        self.bindings, self.trail, self.memo, self.visited = snap


class MatchingMixin:
    """Mixin for matching logic in PatternDetector.

    This implementation uses a recursive depth-first search with backtracking and
    memoization to handle complex subgraph patterns, including wildcards,
    alternative paths, and commutative operators.
    """

    def _get_backend(self) -> Any:
        """Access the underlying backend parser."""
        if hasattr(self, "detector"):
            return self.detector._backend
        return getattr(self, "_backend", None)

    def _build_nx_graph(self: PatternDetector) -> nx.DiGraph:
        """Lazily build a NetworkX representation of the graph."""
        if self._nx_graph is None:
            g = nx.DiGraph()
            for n in self._nodes:
                name = getattr(n, "name", f"node_{id(n)}")
                g.add_node(name, proto=n)
                for inp in n.input:
                    parent = self._output_to_node.get(inp)
                    if parent:
                        p_name = getattr(parent, "name", f"node_{id(parent)}")
                        g.add_edge(p_name, name)
            self._nx_graph = g
        return self._nx_graph

    def _resolve(self: PatternDetector, node: str | Any | None) -> Any | None:
        """Resolve a node name or proto to a Node instance."""
        if node is None or not isinstance(node, str):
            return node
        for n in self._nodes:
            if getattr(n, "name", "") == node:
                return n
        msg = f"Node not found in graph: '{node}'"
        raise ValueError(msg)

    def _descendant_nodes(self: PatternDetector) -> list[Any]:
        """Collect all descendants of self.start (inclusive)."""
        if self.start is None:
            msg = "start_node must be set to collect descendants"
            raise ValueError(msg)
        g = self._build_nx_graph()
        name = getattr(self.start, "name", f"node_{id(self.start)}")
        desc_names = nx.descendants(g, name) | {name}
        return [g.nodes[dn]["proto"] for dn in desc_names]

    def _match_recursive(
        self: PatternDetector,
        node: Any,
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
        if not self._check_attrs(node, pattern) or not self._check_shape_dtype(
            node, pattern
        ):
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

    def _match_any_of(
        self: PatternDetector, node: Any, pattern: Pattern, ctx: MatchContext
    ) -> bool:
        """Match if any of the alternatives match."""
        for alt in pattern._alternatives:
            snap = ctx.snapshot()
            if self._match_recursive(node, alt, ctx):
                if pattern._capture:
                    if pattern._capture in ctx.bindings and id(
                        ctx.bindings[pattern._capture]
                    ) != id(node):
                        ctx.restore(snap)
                        continue
                    ctx.bindings[pattern._capture] = node
                ctx.memo[id(pattern)] = id(node)
                return True
            ctx.restore(snap)
        return False

    def _match_const_pattern(
        self: PatternDetector, node: Any, pattern: Pattern, ctx: MatchContext
    ) -> bool:
        """Match against a constant value."""
        if self._match_init_const_node(node, pattern.value):
            ctx.memo[id(pattern)] = id(node)
            return True
        return False

    def _match_wildcard(
        self: PatternDetector, node: Any, pattern: Pattern, ctx: MatchContext
    ) -> bool:
        """Match any node, subject to shape/dtype constraints."""
        if not self._check_shape_dtype(node, pattern):
            return False
        return self._finalize_match(node, pattern, ctx)

    def _match_ordered(
        self: PatternDetector,
        node: Any,
        parents: list[Any | None],
        pattern: Pattern,
        ctx: MatchContext,
    ) -> bool:
        """Match inputs in the exact order specified."""
        if len(parents) < len(pattern.inputs):
            return False

        snap = ctx.snapshot()
        name = getattr(node, "name", f"node_{id(node)}")
        ctx.visited.add(name)
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
        node: Any,
        parents: list[Any | None],
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

        name = getattr(node, "name", f"node_{id(node)}")
        for perm in itertools.permutations(actual_parents, len(non_const_pats)):
            snap = ctx.snapshot()
            ctx.visited.add(name)
            if all(
                self._match_recursive(p, pat, ctx)
                for p, pat in zip(perm, non_const_pats)
            ):
                return True
            ctx.restore(snap)
        return False

    def _finalize_match(
        self: PatternDetector, node: Any, pattern: Pattern, ctx: MatchContext
    ) -> bool:
        """Record the successful match and update bindings/trail."""
        if not any(n is node for n in ctx.trail):
            ctx.trail.append(node)
        if pattern._capture:
            ctx.bindings[pattern._capture] = node
        ctx.memo[id(pattern)] = id(node)
        return True

    def _get_parent_nodes(
        self: PatternDetector, node: Any, ctx: MatchContext
    ) -> list[Any | None]:
        """Get the parent nodes for a given node, filtering out visited ones."""
        parents: list[Any | None] = []
        for inp in node.input:
            if not inp or inp in self._tensor_map:
                parents.append(None)
                continue
            parent = self._output_to_node.get(inp)
            p_name = getattr(parent, "name", f"node_{id(parent)}") if parent else None
            if parent and p_name not in ctx.visited:
                parents.append(parent)
            else:
                parents.append(None)
        return parents

    def _match_init_const_node(
        self: PatternDetector, node: Any, value: object, used: set[str] | None = None
    ) -> bool:
        """Match if any input to the node is an initializer/constant with the given value."""
        try:
            for i in range(len(node.input)):
                if self._match_init_const_at_pos(node, i, value, used or set()):
                    return True

            # Backend-specific constant node check
            backend = self._get_backend()
            if backend and backend.is_constant_node(node):
                val = backend.get_constant_value(node)
                if val is not None:
                    return bool(np.allclose(val, cast(Any, value), atol=1e-3))
        except (ValueError, TypeError, AttributeError):
            pass
        return False

    def _match_init_const_at_pos(
        self: PatternDetector, node: Any, pos: int, value: object, used: set[str]
    ) -> bool:
        """Match a constant/initializer at a specific input position."""
        if pos >= len(node.input):
            return False
        inp = node.input[pos]
        if not inp or inp in used:
            return False

        try:
            # Case 1: Initializer/Tensor map
            if inp in self._tensor_map:
                if np.allclose(self._tensor_map[inp], cast(Any, value), atol=1e-3):
                    used.add(inp)
                    return True
            # Case 2: Constant node output
            parent = self._output_to_node.get(inp)
            backend = self._get_backend()
            if parent and backend and backend.is_constant_node(parent):
                val = backend.get_constant_value(parent)
                print(f"DEBUG: Found constant parent {parent.name} with value {val}")
                if val is not None and np.allclose(val, cast(Any, value), atol=1e-3):
                    used.add(inp)
                    return True
            else:
                print(
                    f"DEBUG: No constant parent found for input {inp} (parent={parent}, backend={backend})"
                )
        except (ValueError, TypeError, AttributeError):
            pass
        return False

    def _check_attrs(self: PatternDetector, node: Any, pattern: Pattern) -> bool:
        """Check if node attributes satisfy pattern constraints."""
        if not pattern._constraints:
            return True

        backend = self._get_backend()
        if backend:
            attrs = backend.get_node_attrs(node)
        else:
            # Fallback to generic extractor if no backend (e.g. shim)
            from neuron_toolkit.query.core import _get_node_attrs

            attrs = _get_node_attrs(node)

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

    def _check_shape_dtype(self: PatternDetector, node: Any, pattern: Pattern) -> bool:
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
