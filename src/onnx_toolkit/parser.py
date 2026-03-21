"""
onnx_toolkit.parser
====================
ONNXParser — load an ONNX model and expose it for querying, pattern
detection, and rewriting.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import List, Optional, Set, Union

import onnx
from onnx import ModelProto, numpy_helper
from onnx.onnx_pb import NodeProto

from ._types import TensorMap
from ._utils import _build_shape_info, ShapeInfo
from .pattern import Pattern, PatternDetector, MatchResult
from .query import ONNXQuery

log = logging.getLogger("onnx_toolkit")


class ONNXParser:
    """
    Load an ONNX model and expose it for querying, pattern detection,
    and graph rewriting.

    Parameters
    ----------
    onnx_path      : str   Path to the .onnx file.
    infer_shapes   : bool  Run onnx.shape_inference on load (default True).
                           Required for .with_output_rank() / .with_dtype()
                           pattern constraints to work.

    Example
    -------
    >>> parser = ONNXParser("model.onnx")
    >>> convs  = parser.find().find_by_op_type("Conv").has_params()
    >>> print(convs)
    >>> print(parser.summary())
    """

    def __init__(self, onnx_path: str, *, infer_shapes: bool = True) -> None:
        log.info("Loading ONNX model from %r", onnx_path)
        self.model: ModelProto = onnx.load(onnx_path)

        if infer_shapes:
            try:
                self.model = onnx.shape_inference.infer_shapes(self.model)
                log.debug("Shape inference completed.")
            except Exception as exc:
                log.warning("Shape inference failed: %s", exc)

        self.nodes: List[NodeProto] = list(self.model.graph.node)
        self.tensor_map: TensorMap  = {
            t.name: numpy_helper.to_array(t)
            for t in self.model.graph.initializer
        }
        self.graph_inputs:  Set[str] = {i.name for i in self.model.graph.input}
        self.graph_outputs: Set[str] = {o.name for o in self.model.graph.output}
        self.shape_info: ShapeInfo   = _build_shape_info(self.model)

        log.info(
            "Model loaded: %d nodes, %d tensors, %d inputs, %d outputs, "
            "%d shape annotations",
            len(self.nodes), len(self.tensor_map),
            len(self.graph_inputs), len(self.graph_outputs),
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
        pattern:    Pattern,
        start_node: Optional[Union[str, NodeProto]] = None,
        end_node:   Optional[Union[str, NodeProto]] = None,
    ) -> Optional[MatchResult]:
        """
        Create a PatternDetector bound to this model and call match().

        Returns a :class:`MatchResult` on success, ``None`` on failure.
        """
        from ._utils import _GraphShim
        shim = _GraphShim(self.nodes, self.tensor_map, self.shape_info)
        det  = PatternDetector(shim, start_node=start_node, end_node=end_node)
        return det.match(pattern)

    def rewriter(self) -> "GraphRewriter":  # type: ignore[name-defined]
        """Return a :class:`GraphRewriter` bound to this model."""
        from .rewriter import GraphRewriter
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
