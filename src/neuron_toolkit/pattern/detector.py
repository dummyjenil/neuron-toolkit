from __future__ import annotations

import networkx as nx
from onnx import ModelProto, numpy_helper
from onnx.onnx_pb import NodeProto

from neuron_toolkit._utils import ShapeInfo, _GraphShim
from neuron_toolkit.pattern._matcher import MatchContext, MatchingMixin
from neuron_toolkit.pattern.dsl import Pattern
from neuron_toolkit.pattern.models import MatchResult


class PatternDetector(MatchingMixin):
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

    def match(self, pattern: Pattern) -> MatchResult | None:
        """Attempt to match *pattern* starting from self.start."""
        if self.start is None:
            return None

        ctx = MatchContext(detector=self)
        if not self._match_recursive(self.start, pattern, ctx):
            return None

        terminal = self.end if self.end is not None else self.start
        return MatchResult(
            start=self.start,
            end=terminal,
            nodes=ctx.trail,
            bindings=ctx.bindings,
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
