"""State management and matching logic for PatternDetector."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np

from neuron_toolkit._utils import ShapeInfo
from neuron_toolkit.pattern.dsl import _ANY_OF, _CONST_PAT, _WILDCARD

if TYPE_CHECKING:
    from neuron_toolkit.pattern.detector import PatternDetector
    from neuron_toolkit.pattern.dsl import Pattern

log = logging.getLogger("neuron_toolkit.detect")


@dataclass
class MatchContext:
    """State management for a single pattern matching attempt."""

    detector: PatternDetector
    bindings: dict[str, object] = field(default_factory=dict)
    trail: list[object] = field(default_factory=list)
    # Maps id(Pattern) -> id(Node) to enforce referential consistency
    memo: dict[int, int] = field(default_factory=dict)
    visited: set[str] = field(default_factory=set)

    def snapshot(
        self,
    ) -> tuple[dict[str, object], list[object], dict[int, int], set[str]]:
        """Take a snapshot of the current state for backtracking."""
        return (
            dict(self.bindings),
            list(self.trail),
            dict(self.memo),
            set(self.visited),
        )

    def restore(
        self,
        snap: tuple[dict[str, object], list[object], dict[int, int], set[str]],
    ) -> None:
        """Restore state from a snapshot."""
        self.bindings, self.trail, self.memo, self.visited = snap


class MatchingMixin:
    """Mixin for matching logic in PatternDetector.

    This implementation uses a recursive depth-first search with backtracking to
    handle complex subgraph patterns, including wildcards, alternative paths,
    and commutative operators.
    """

    _nodes: Sequence[object]
    _tensor_map: Mapping[str, object]
    _shape_info: ShapeInfo
    _nx_graph: nx.DiGraph | None
    output_to_node: dict[str, object]
    start: object | None
    end: object | None

    @property
    def backend(self) -> object | None:
        """Return the backend parser."""
        return self._get_backend()

    def _get_backend(self) -> object | None:
        """Access the underlying backend parser."""
        if hasattr(self, "detector"):
            return cast(object | None, cast(Any, self).detector.backend)
        return getattr(self, "_backend", None)

    def _build_nx_graph(self) -> nx.DiGraph:
        """Lazily build a NetworkX representation of the graph."""
        if self._nx_graph is None:
            g = nx.DiGraph()
            for n in self._nodes:
                name = getattr(n, "name", f"node_{id(n)}")
                g.add_node(name, proto=n)
                node_inputs = getattr(n, "input", [])
                for inp in node_inputs:
                    parent = self.output_to_node.get(inp)
                    if parent:
                        p_name = getattr(parent, "name", f"node_{id(parent)}")
                        g.add_edge(p_name, name)
            self._nx_graph = g
        return self._nx_graph

    def _resolve(self, node: str | object | None) -> object | None:
        """Resolve a node name or proto to a Node instance."""
        if node is None or not isinstance(node, str):
            return node
        for n in self._nodes:
            if getattr(n, "name", "") == node:
                return n
        msg = f"Node not found in graph: '{node}'"
        raise ValueError(msg)

    def _descendant_nodes(self) -> list[object]:
        """Collect all descendants of self.start (inclusive)."""
        if self.start is None:
            msg = "start_node must be set to collect descendants"
            raise ValueError(msg)
        g = self._build_nx_graph()
        name = getattr(self.start, "name", f"node_{id(self.start)}")
        desc_names = nx.descendants(g, name) | {name}
        return [g.nodes[dn]["proto"] for dn in desc_names]

    def _match_recursive(  # noqa: PLR0911
        self,
        node: object,
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
        node_op_type = getattr(node, "op_type", None)
        if pattern.op_type is not None and node_op_type != pattern.op_type:
            return False
        if not self._check_attrs(node, pattern) or not self._check_shape_dtype(
            node, pattern
        ):
            return False

        # 5. Input Matching
        if not pattern.inputs:
            return self._finalize_match(node, pattern, ctx)

        parents = self._get_parent_nodes(node, ctx)
        if node_op_type in {"Add", "Mul"}:
            ok = self._match_commutative(node, parents, pattern, ctx)
        else:
            ok = self._match_ordered(node, parents, pattern, ctx)

        if ok:
            return self._finalize_match(node, pattern, ctx)
        return False

    def _match_any_of(self, node: object, pattern: Pattern, ctx: MatchContext) -> bool:
        """Match if any of the alternatives match."""
        for alt in pattern.alternatives:
            snap = ctx.snapshot()
            if self._match_recursive(node, alt, ctx):
                if pattern.capture_name:
                    if pattern.capture_name in ctx.bindings and id(
                        ctx.bindings[pattern.capture_name]
                    ) != id(node):
                        ctx.restore(snap)
                        continue
                    ctx.bindings[pattern.capture_name] = node
                ctx.memo[id(pattern)] = id(node)
                return True
            ctx.restore(snap)
        return False

    def _match_const_pattern(
        self, node: object, pattern: Pattern, ctx: MatchContext
    ) -> bool:
        """Match against a constant value."""
        if self._match_init_const_node(node, pattern.value):
            ctx.memo[id(pattern)] = id(node)
            return True
        return False

    def _match_wildcard(
        self, node: object, pattern: Pattern, ctx: MatchContext
    ) -> bool:
        """Match any node, subject to shape/dtype constraints."""
        if not self._check_shape_dtype(node, pattern):
            return False
        return self._finalize_match(node, pattern, ctx)

    def _match_ordered(
        self,
        node: object,
        parents: list[object | None],
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
        self,
        node: object,
        parents: list[object | None],
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
        if len(actual_parents) > 8:  # noqa: PLR2004
            log.warning("Too many parents for commutative match, limiting search")
            actual_parents = actual_parents[:8]

        name = getattr(node, "name", f"node_{id(node)}")
        for perm in itertools.permutations(actual_parents, len(non_const_pats)):
            snap = ctx.snapshot()
            ctx.visited.add(name)
            if all(
                self._match_recursive(p, pat, ctx)
                for p, pat in zip(perm, non_const_pats, strict=False)
            ):
                return True
            ctx.restore(snap)
        return False

    def _finalize_match(
        self, node: object, pattern: Pattern, ctx: MatchContext
    ) -> bool:
        """Record the successful match and update bindings/trail."""
        if not any(n is node for n in ctx.trail):
            ctx.trail.append(node)
        if pattern.capture_name:
            ctx.bindings[pattern.capture_name] = node
        ctx.memo[id(pattern)] = id(node)
        return True

    def _get_parent_nodes(self, node: object, ctx: MatchContext) -> list[object | None]:
        """Get the parent nodes for a given node, filtering out visited ones."""
        parents: list[object | None] = []
        node_inputs = getattr(node, "input", [])
        for inp in node_inputs:
            if not inp or inp in self._tensor_map:
                parents.append(None)
                continue
            parent = self.output_to_node.get(inp)
            p_name = getattr(parent, "name", f"node_{id(parent)}") if parent else None
            if parent and p_name not in ctx.visited:
                parents.append(parent)
            else:
                parents.append(None)
        return parents

    def _match_init_const_node(
        self,
        node: object,
        value: object,
        used: set[str] | None = None,
    ) -> bool:
        """Match if any input to the node is an initializer/constant."""
        try:
            node_inputs = getattr(node, "input", [])
            for i in range(len(node_inputs)):
                if self._match_init_const_at_pos(node, i, value, used or set()):
                    return True

            # Backend-specific constant node check
            backend = self._get_backend()
            if (
                backend
                and hasattr(backend, "is_constant_node")
                and hasattr(backend, "get_constant_value")
                and cast(Any, backend).is_constant_node(node)
            ):
                val = cast(Any, backend).get_constant_value(node)
                if val is not None:
                    return bool(np.allclose(val, cast(Any, value), atol=1e-3))
        except (ValueError, TypeError, AttributeError):
            pass
        return False

    def _match_init_const_at_pos(
        self,
        node: object,
        pos: int,
        value: object,
        used: set[str],
    ) -> bool:
        """Match a constant/initializer at a specific input position."""
        node_inputs = getattr(node, "input", [])
        if pos >= len(node_inputs):
            return False
        inp = node_inputs[pos]
        if not inp or inp in used:
            return False

        try:
            # Case 1: Initializer/Tensor map
            if inp in self._tensor_map and np.allclose(
                cast(Any, self._tensor_map[inp]), cast(Any, value), atol=1e-3
            ):
                used.add(inp)
                return True
            # Case 2: Constant node output
            parent = self.output_to_node.get(inp)
            backend = self._get_backend()
            if (
                parent
                and backend
                and hasattr(backend, "is_constant_node")
                and hasattr(backend, "get_constant_value")
            ):
                if cast(Any, backend).is_constant_node(parent):
                    val = cast(Any, backend).get_constant_value(parent)
                    print(
                        f"DEBUG: Found constant parent {getattr(parent, 'name', '')} "
                        f"with value {val}"
                    )
                    if val is not None and np.allclose(
                        val, cast(Any, value), atol=1e-3
                    ):
                        used.add(inp)
                        return True
            else:
                print(
                    f"DEBUG: No constant parent found for input {inp} "
                    f"(parent={parent}, backend={backend})"
                )
        except (ValueError, TypeError, AttributeError):
            pass
        return False

    def _check_attrs(self, node: object, pattern: Pattern) -> bool:
        """Check if node attributes satisfy pattern constraints."""
        if not pattern.constraints:
            return True

        backend = self._get_backend()
        if backend and hasattr(backend, "get_node_attrs"):
            attrs = cast(Any, backend).get_node_attrs(node)
        else:
            # Fallback to generic extractor if no backend (e.g. shim)
            from neuron_toolkit.query.core import _get_node_attrs  # noqa: PLC0415

            attrs = _get_node_attrs(node)

        for attr_name, expected in pattern.constraints.items():
            if attr_name not in attrs:
                return False
            actual = attrs[attr_name]
            if callable(expected):
                if not cast("Callable[[Any], bool]", expected)(actual):
                    return False
            elif isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
                if not np.array_equal(cast(Any, actual), cast(Any, expected)):
                    return False
            elif actual != expected:
                return False
        return True

    def _check_shape_dtype(self, node: object, pattern: Pattern) -> bool:
        """Check if node output shape and dtype satisfy constraints."""
        if pattern.rank is None and pattern.dtype_str is None:
            return True
        node_outputs = getattr(node, "output", [])
        if not node_outputs:
            return False
        rank, dtype = self._shape_info.get(node_outputs[0], (None, None))
        if pattern.rank is not None and rank != pattern.rank:
            return False
        return pattern.dtype_str is None or dtype == pattern.dtype_str
