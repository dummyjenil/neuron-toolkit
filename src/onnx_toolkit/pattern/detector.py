from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from typing import Any, cast

import networkx as nx
import numpy as np
from onnx import ModelProto, numpy_helper
from onnx.onnx_pb import NodeProto

from onnx_toolkit._utils import ShapeInfo, _GraphShim, _node_attrs
from onnx_toolkit.pattern.dsl import _ANY_OF, _CONST_PAT, _WILDCARD, Pattern
from onnx_toolkit.pattern.models import MatchResult

log = logging.getLogger("onnx_toolkit.detect")


class PatternDetector:
    """Match a Pattern against a subgraph of an ONNX model."""

    def __init__(
        self,
        model: ModelProto | _GraphShim,
        start_node: str | NodeProto | None = None,
        end_node: str | NodeProto | None = None,
    ) -> None:
        """Initialize PatternDetector with model and optional start/end nodes."""
        if isinstance(model, _GraphShim):
            self._nodes = model.nodes
            self._tensor_map = model.tensor_map
            self._shape_info: ShapeInfo = model.shape_info
        else:
            self._nodes = list(model.graph.node)
            self._tensor_map = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
            self._shape_info = {}

        self._output_to_node: dict[str, NodeProto] = {
            out: n for n in self._nodes for out in n.output
        }
        self.start = self._resolve(start_node)
        self.end = self._resolve(end_node)
        self._nx_graph: nx.DiGraph | None = None

    def _build_nx_graph(self) -> nx.DiGraph:
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

    def match(self, pattern: Pattern) -> MatchResult | None:
        """Attempt to match *pattern* starting from self.start."""
        if self.start is None:
            return None
        bindings: dict[str, NodeProto] = {}
        # Track pattern identity to node identity to enforce referential consistency
        pat_to_node: dict[int, int] = {}
        visited_nodes: list[NodeProto] = []

        if not self._dfs(self.start, pattern, frozenset(), bindings, visited_nodes, pat_to_node):
            return None

        terminal = self.end if self.end is not None else self.start
        return MatchResult(
            start=self.start,
            end=terminal,
            nodes=visited_nodes,
            bindings=bindings,
        )

    def find_all(self, pattern: Pattern) -> list[MatchResult]:
        """Find all matches for *pattern* in the subgraph reachable from self.start."""
        candidates = self._descendant_nodes() if self.start is not None else self._nodes
        shim = _GraphShim(self._nodes, self._tensor_map, self._shape_info)
        results: list[MatchResult] = []
        for node in candidates:
            if node is self.end:
                continue
            det = PatternDetector(shim, start_node=node, end_node=self.end)
            det._output_to_node = self._output_to_node
            r = det.match(pattern)
            if r is not None:
                results.append(r)
        return results

    def _resolve(self, node: str | NodeProto | None) -> NodeProto | None:
        if node is None or isinstance(node, NodeProto):
            return node
        for n in self._nodes:
            if n.name == node:
                return n
        msg = f"Node not found in graph: '{node}'"
        raise ValueError(msg)

    def _descendant_nodes(self) -> list[NodeProto]:
        """Collect all descendants of self.start (inclusive)."""
        if self.start is None:
            msg = "start_node must be set to collect descendants"
            raise ValueError(msg)
        g = self._build_nx_graph()
        desc_names = nx.descendants(g, self.start.name) | {self.start.name}
        return [g.nodes[name]["proto"] for name in desc_names]

    def _dfs(
        self,
        node: NodeProto,
        pattern: Pattern,
        visited: frozenset[str],
        bindings: dict[str, NodeProto],
        trail: list[NodeProto],
        pat_to_node: dict[int, int],
    ) -> bool:
        # Enforce referential consistency: if this pattern object was already matched,
        # it MUST match the exact same node object.
        pat_id = id(pattern)
        if pat_id in pat_to_node:
            return pat_to_node[pat_id] == id(node)

        if self.end is not None and node is self.end:
            return True

        if pattern.op_type == _ANY_OF:
            for alt in pattern._alternatives:
                snap_b = dict(bindings)
                snap_t = len(trail)
                snap_p = dict(pat_to_node)
                if self._dfs(node, alt, visited, bindings, trail, pat_to_node):
                    if pattern._capture:
                        # Consistency check for capture
                        if pattern._capture in bindings and id(bindings[pattern._capture]) != id(
                            node
                        ):
                            # Backtrack
                            pass
                        else:
                            bindings[pattern._capture] = node
                            pat_to_node[pat_id] = id(node)
                            return True
                    else:
                        pat_to_node[pat_id] = id(node)
                        return True
                bindings.clear()
                bindings.update(snap_b)
                del trail[snap_t:]
                pat_to_node.clear()
                pat_to_node.update(snap_p)
            return False

        if pattern.op_type == _CONST_PAT:
            res = self._match_const(node, pattern.value)
            if res:
                pat_to_node[pat_id] = id(node)
            return res

        if pattern.op_type == _WILDCARD:
            if not self._check_shape_dtype(node, pattern):
                return False
            self._record(node, pattern, bindings, trail)
            pat_to_node[pat_id] = id(node)
            return True

        if pattern.op_type is not None and node.op_type != pattern.op_type:
            return False

        if not self._check_attrs(node, pattern) or not self._check_shape_dtype(node, pattern):
            return False

        if not pattern.inputs:
            self._record(node, pattern, bindings, trail)
            pat_to_node[pat_id] = id(node)
            return True

        parents = self._parents_with_placeholders(node, visited)

        if node.op_type in ("Add", "Mul"):
            # Commutative: separate constants and nodes
            non_const_pats: list[Pattern] = []
            used_inits: set[str] = set()
            for pat in pattern.inputs:
                if pat.op_type == _CONST_PAT:
                    if not self._match_init_const(node, pat.value, used_inits):
                        return False
                else:
                    non_const_pats.append(pat)

            # Match remaining node patterns against remaining node parents
            actual_parents = [p for p in parents if p is not None]
            if len(actual_parents) < len(non_const_pats):
                return False

            ok = self._try_commutative(
                actual_parents, non_const_pats, visited | {node.name}, bindings, trail, pat_to_node
            )
        else:
            # Ordered: match patterns one-by-one in position
            if len(parents) < len(pattern.inputs):
                return False

            ok = True
            new_visited = visited | {node.name}
            used_inits = set()

            for i, pat in enumerate(pattern.inputs):
                if pat.op_type == _CONST_PAT:
                    # Must match the input at this specific position
                    if not self._match_init_const_at_pos(node, i, pat.value, used_inits):
                        ok = False
                        break
                else:
                    # Must be a node at this position
                    parent = parents[i]
                    if parent is None:
                        ok = False
                        break
                    if not self._dfs(parent, pat, new_visited, bindings, trail, pat_to_node):
                        ok = False
                        break

        if ok:
            self._record(node, pattern, bindings, trail)
            pat_to_node[pat_id] = id(node)
        return ok

    def _match_const(self, node: NodeProto, value: object) -> bool:
        try:
            inits = [self._tensor_map[inp] for inp in node.input if inp and inp in self._tensor_map]
            if inits:
                return any(np.allclose(a, cast(Any, value), atol=1e-3) for a in inits)
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value":
                        return bool(np.allclose(numpy_helper.to_array(attr.t), cast(Any, value), atol=1e-3))
        except (ValueError, TypeError):
            # Handle shape mismatch or broadcasting errors gracefully
            return False
        return False

    def _match_init_const(self, node: NodeProto, value: object, used: set[str]) -> bool:
        try:
            for inp in node.input:
                if not inp or inp in used:
                    continue
                # Case 1: Initializer
                if inp in self._tensor_map:
                    if np.allclose(self._tensor_map[inp], cast(Any, value), atol=1e-3):
                        used.add(inp)
                        return True
                # Case 2: Constant node
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

    def _check_attrs(self, node: NodeProto, pattern: Pattern) -> bool:
        if not pattern._constraints:
            return True
        attrs = _node_attrs(node)
        for attr_name, expected in pattern._constraints.items():
            if attr_name not in attrs:
                return False
            actual = attrs[attr_name]
            if callable(expected):
                func = cast("Callable[[Any], bool]", expected)
                if not func(actual):
                    return False
            elif isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
                if not np.array_equal(actual, cast(Any, expected)):
                    return False
            elif actual != expected:
                return False
        return True

    def _check_shape_dtype(self, node: NodeProto, pattern: Pattern) -> bool:
        if pattern._rank is None and pattern._dtype is None:
            return True
        if not node.output:
            return False
        rank, dtype = self._shape_info.get(node.output[0], (None, None))
        if pattern._rank is not None and rank != pattern._rank:
            return False
        return not (pattern._dtype is not None and dtype != pattern._dtype)

    def _match_init_const_at_pos(
        self, node: NodeProto, pos: int, value: object, used: set[str]
    ) -> bool:
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
            # Case 2: Constant node
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

    def _parents_with_placeholders(
        self, node: NodeProto, visited: frozenset[str]
    ) -> list[NodeProto | None]:
        parents: list[NodeProto | None] = []
        for inp in node.input:
            if not inp or inp in self._tensor_map:
                parents.append(None)
                continue
            parent = self._output_to_node.get(inp)
            if parent and parent.name not in visited:
                parents.append(parent)
            else:
                parents.append(None)
        return parents

    def _try_commutative(
        self,
        parents: list[NodeProto],
        patterns: list[Pattern],
        visited: frozenset[str],
        bindings: dict[str, NodeProto],
        trail: list[NodeProto],
        pat_to_node: dict[int, int],
    ) -> bool:
        # Limit permutations to avoid DoS on nodes with many inputs
        if len(parents) > 8:
            log.warning("Too many parents for commutative match, limiting to first 8")
            parents = parents[:8]

        for perm in itertools.permutations(parents, min(len(patterns), len(parents))):
            snap_b = dict(bindings)
            snap_t = len(trail)
            snap_p = dict(pat_to_node)
            if all(
                self._dfs(p, pat, visited | {p.name}, bindings, trail, pat_to_node)
                for p, pat in zip(perm, patterns)
            ):
                return True
            bindings.clear()
            bindings.update(snap_b)
            del trail[snap_t:]
            pat_to_node.clear()
            pat_to_node.update(snap_p)
        return False

    @staticmethod
    def _record(
        node: NodeProto,
        pattern: Pattern,
        bindings: dict[str, NodeProto],
        trail: list[NodeProto],
    ) -> None:
        if not any(n is node for n in trail):
            trail.append(node)
        if pattern._capture:
            # We enforce consistency in _dfs, but record here too
            bindings[pattern._capture] = node
