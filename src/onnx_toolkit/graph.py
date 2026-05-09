"""onnx_toolkit.graph.

Unified entry point for ONNX graph analysis, querying, and transformation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from onnx_toolkit.parser import ONNXParser

if TYPE_CHECKING:
    from onnx.onnx_pb import NodeProto

    from onnx_toolkit.pattern import MatchResult, Pattern
    from onnx_toolkit.query import ONNXQuery
    from onnx_toolkit.rewriter import GraphRewriter

log = logging.getLogger("onnx_toolkit.graph")


class ONNXGraph(ONNXParser):
    """Unified interface for an ONNX model graph."""

    def query(self) -> ONNXQuery:
        """Alias for find() — return a query over all nodes."""
        return self.find()

    def match(self, pattern: Pattern) -> MatchResult | None:
        """Find the first occurrence of *pattern* in the graph."""
        results = self.findall(pattern)
        return results[0] if results else None

    def findall(self, pattern: Pattern) -> list[MatchResult]:
        """Find all occurrences of *pattern* in the graph."""
        from onnx_toolkit.pattern import PatternDetector

        det = PatternDetector(self._shim())
        return det.find_all(pattern)

    def replace(
        self,
        pattern: Pattern,
        new_op: str,
        name: str | None = None,
        **attrs: Any,
    ) -> GraphRewriter:
        """Replace all matches of *pattern* with a new operator.

        This is a fluent shortcut for finding all matches and applying a rewrite.
        """
        rewriter = self.rewriter()
        matches = self.findall(pattern)
        for m in matches:
            rewriter.replace_from_result(m, new_op, name=name, **attrs)
        return rewriter

    def _shim(self) -> Any:
        from onnx_toolkit._utils import _GraphShim

        return _GraphShim(self.nodes, self.tensor_map, self.shape_info)

    @property
    def passes(self) -> GraphPasses:
        """Return a namespace for common graph optimization passes."""
        return GraphPasses(self)


class GraphPasses:
    """Namespace for common graph optimization passes."""

    def __init__(self, graph: ONNXGraph) -> None:
        self.graph = graph

    def fuse_conv_bn(self) -> GraphRewriter:
        """Example pass: Fuse Conv and BatchNormalization."""
        from onnx_toolkit.pattern import Pattern

        conv = Pattern.op("Conv").capture("conv")
        bn = Pattern.op("BatchNormalization", conv).capture("bn")
        return self.graph.replace(bn, "Conv")  # Placeholder for real fusion logic
