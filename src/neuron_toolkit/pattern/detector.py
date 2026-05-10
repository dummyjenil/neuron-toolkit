from __future__ import annotations

from typing import Any

import networkx as nx

from neuron_toolkit._utils import ShapeInfo, _GraphShim
from neuron_toolkit.pattern._matcher import MatchContext, MatchingMixin
from neuron_toolkit.pattern.dsl import Pattern
from neuron_toolkit.pattern.models import MatchResult


class PatternDetector(MatchingMixin):
    """Match a Pattern against a subgraph of a model (ONNX, TFLite, etc.)."""

    def __init__(
        self,
        model: Any,
        start_node: Any | str | None = None,
        end_node: Any | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize PatternDetector with model and optional start/end nodes."""
        if isinstance(model, _GraphShim):
            self._nodes = model.nodes
            self._tensor_map = model.tensor_map
            self._shape_info: ShapeInfo = model.shape_info
            self._backend = model.backend
        elif hasattr(model, "graph") and hasattr(
            model.graph, "node"
        ):  # ONNX ModelProto
            from neuron_toolkit.backends.onnx.parser import ONNXParser

            self._backend = ONNXParser(model, **kwargs)
            self._nodes = self._backend.nodes
            self._tensor_map = self._backend.tensor_map
            self._shape_info = self._backend.shape_info
        elif hasattr(model, "Subgraphs"):  # TFLite Model
            # If it's a TFLite flatbuffer, TFLiteParser currently expects a path.
            # This is a gap: TFLiteParser should support both path and model object.
            # For now, we'll assume it's already wrapped or handle as generic.
            self._backend = model
            self._nodes = getattr(model, "nodes", [])
            self._tensor_map = getattr(model, "tensor_map", {})
            self._shape_info = getattr(model, "shape_info", {})
        else:
            # Assume it's already a backend parser (ONNXParser or TFLiteParser)
            self._backend = model
            self._nodes = getattr(model, "nodes", [])
            self._tensor_map = getattr(model, "tensor_map", {})
            self._shape_info = getattr(model, "shape_info", {})

        self._output_to_node: dict[str, Any] = {
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
        shim = _GraphShim(
            self._nodes, self._tensor_map, self._shape_info, backend=self._backend
        )
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
