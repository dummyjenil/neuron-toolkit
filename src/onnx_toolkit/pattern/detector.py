from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, cast

import networkx as nx
import numpy as np
from onnx import ModelProto, numpy_helper
from onnx.onnx_pb import NodeProto

from onnx_toolkit._utils import ShapeInfo, _GraphShim, _node_attrs
from onnx_toolkit.pattern.dsl import _ANY_OF, _CONST_PAT, _WILDCARD, Pattern
from onnx_toolkit.pattern.models import MatchResult

log_d = logging.getLogger("onnx_toolkit.detect")


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
        visited_nodes: list[NodeProto] = []
        if not self._dfs(self.start, pattern, frozenset(), bindings, visited_nodes):
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
    ) -> bool:
        if self.end is not None and node is self.end:
            return True

        if pattern.op_type == _ANY_OF:
            for alt in pattern._alternatives:
                snap_b = dict(bindings)
                snap_t = len(trail)
                if self._dfs(node, alt, visited, bindings, trail):
                    if pattern._capture:
                        bindings[pattern._capture] = node
                    return True
                bindings.clear()
                bindings.update(snap_b)
                del trail[snap_t:]
            return False

        if pattern.op_type == _CONST_PAT:
            return self._match_const(node, pattern.value)

        if pattern.op_type == _WILDCARD:
            if not self._check_shape_dtype(node, pattern):
                return False
            self._record(node, pattern, bindings, trail)
            return True

        if pattern.op_type is not None and node.op_type != pattern.op_type:
            return False

        if not self._check_attrs(node, pattern) or not self._check_shape_dtype(node, pattern):
            return False

        if not pattern.inputs:
            self._record(node, pattern, bindings, trail)
            return True

        parents = self._parents(node, visited)
        non_const_pats: list[Pattern] = []
        used_inits: set[str] = set()

        for pat in pattern.inputs:
            if pat.op_type == _CONST_PAT:
                if not self._match_init_const(node, pat.value, used_inits):
                    return False
            else:
                non_const_pats.append(pat)

        if len(parents) < len(non_const_pats):
            return False

        new_visited = visited | {node.name}
        if node.op_type in ("Add", "Mul"):
            ok = self._try_commutative(parents, non_const_pats, new_visited, bindings, trail)
        else:
            ok = self._try_ordered(parents, non_const_pats, new_visited, bindings, trail)

        if ok:
            self._record(node, pattern, bindings, trail)
        return ok

    def _match_const(self, node: NodeProto, value: object) -> bool:
        inits = [self._tensor_map[inp] for inp in node.input if inp and inp in self._tensor_map]
        if inits:
            return any(np.allclose(a, value, atol=1e-3) for a in inits)
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    return bool(np.allclose(numpy_helper.to_array(attr.t), value, atol=1e-3))
        return False

    def _match_init_const(self, node: NodeProto, value: object, used: set[str]) -> bool:
        for inp in node.input:
            if (
                inp
                and inp in self._tensor_map
                and inp not in used
                and np.allclose(self._tensor_map[inp], value, atol=1e-3)
            ):
                used.add(inp)
                return True
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

    def _parents(self, node: NodeProto, visited: frozenset[str]) -> list[NodeProto]:
        parents: list[NodeProto] = []
        for inp in node.input:
            if not inp or inp in self._tensor_map:
                continue
            parent = self._output_to_node.get(inp)
            if parent and parent.name not in visited:
                parents.append(parent)
        return parents

    def _try_ordered(
        self,
        parents: list[NodeProto],
        patterns: list[Pattern],
        visited: frozenset[str],
        bindings: dict[str, NodeProto],
        trail: list[NodeProto],
    ) -> bool:
        for parent, pat in zip(parents, patterns):
            snap_b = dict(bindings)
            snap_t = len(trail)
            if not self._dfs(parent, pat, visited | {parent.name}, bindings, trail):
                bindings.clear()
                bindings.update(snap_b)
                del trail[snap_t:]
                return False
        return True

    def _try_commutative(
        self,
        parents: list[NodeProto],
        patterns: list[Pattern],
        visited: frozenset[str],
        bindings: dict[str, NodeProto],
        trail: list[NodeProto],
    ) -> bool:
        for perm in itertools.permutations(parents, len(patterns)):
            snap_b = dict(bindings)
            snap_t = len(trail)
            if all(
                self._dfs(p, pat, visited | {p.name}, bindings, trail)
                for p, pat in zip(perm, patterns)
            ):
                return True
            bindings.clear()
            bindings.update(snap_b)
            del trail[snap_t:]
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
            bindings[pattern._capture] = node
