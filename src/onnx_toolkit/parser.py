"""onnx_toolkit.parser.

ONNXParser — entry point for model analysis.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import onnx
from onnx import ModelProto, numpy_helper
from onnx.onnx_pb import NodeProto

from onnx_toolkit import _GraphShim
from onnx_toolkit._types import TensorMap
from onnx_toolkit._utils import ShapeInfo, _build_shape_info
from onnx_toolkit.pattern import MatchResult, Pattern, PatternDetector
from onnx_toolkit.query import ONNXQuery

if TYPE_CHECKING:
    from onnx_toolkit.rewriter import GraphRewriter

log = logging.getLogger("onnx_toolkit")


class ONNXParser:
    """Load an ONNX model and expose it for querying and pattern matching."""

    def __init__(self, onnx_path: str, *, infer_shapes: bool = True) -> None:
        """Initialize the ONNXParser with a model file."""
        log.info("Loading ONNX model from %r", onnx_path)
        self.model: ModelProto = onnx.load(onnx_path)

        if infer_shapes:
            try:
                self.model = onnx.shape_inference.infer_shapes(self.model)
                log.debug("Shape inference completed.")
            except Exception as exc:
                log.warning("Shape inference failed: %s", exc)

        self.nodes: list[NodeProto] = list(self.model.graph.node)
        self.tensor_map: TensorMap = {
            t.name: numpy_helper.to_array(t) for t in self.model.graph.initializer
        }
        self.graph_inputs: set[str] = {i.name for i in self.model.graph.input}
        self.graph_outputs: set[str] = {o.name for o in self.model.graph.output}
        self.shape_info: ShapeInfo = _build_shape_info(self.model)

        log.info(
            "Model loaded: %d nodes, %d tensors, %d inputs, %d outputs, %d shape annotations",
            len(self.nodes),
            len(self.tensor_map),
            len(self.graph_inputs),
            len(self.graph_outputs),
            len(self.shape_info),
        )

    # ------------------------------------------------------------------

    def find(self) -> ONNXQuery:
        """Return an ONNXQuery over **all** nodes in the graph."""
        return ONNXQuery(
            list(self.nodes),
            self.tensor_map,
            self.nodes,
            self.graph_inputs,
            self.graph_outputs,
            self.shape_info,
        )

    def pattern_detect(
        self,
        pattern: Pattern,
        start_node: str | NodeProto | None = None,
        end_node: str | NodeProto | None = None,
    ) -> MatchResult | None:
        """Create a PatternDetector bound to this model and call match()."""
        shim = _GraphShim(self.nodes, self.tensor_map, self.shape_info)
        det = PatternDetector(shim, start_node=start_node, end_node=end_node)
        return det.match(pattern)

    def rewriter(self) -> GraphRewriter:
        """Return a :class:`GraphRewriter` bound to this model."""
        from onnx_toolkit.rewriter import GraphRewriter

        return GraphRewriter(self)

    def summary(self) -> str:
        """Return a human-readable summary of the model."""
        op_counts = Counter(n.op_type for n in self.nodes)
        lines = [
            "ONNX model summary",
            f"  Nodes        : {len(self.nodes)}",
            f"  Tensors      : {len(self.tensor_map)}",
            f"  Graph inputs : {len(self.graph_inputs)}",
            f"  Graph outputs: {len(self.graph_outputs)}",
            f"  Shape info   : {len(self.shape_info)} annotated tensors",
            f"  Op types ({len(op_counts)}):",
        ]
        for op, cnt in op_counts.most_common():
            lines.append(f"    {op:<24} {cnt:>5}")
        return "\n".join(lines)
