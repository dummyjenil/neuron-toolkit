"""
onnx_toolkit.parser
====================
ONNXParser: load an ONNX model and expose it for querying and pattern
detection.

Logging
-------
  onnx_toolkit  – top-level / parser messages
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import List, Optional, Set, Union

import onnx
from onnx import ModelProto, numpy_helper
from onnx.onnx_pb import NodeProto

from ._types import TensorMap
from .pattern import Pattern, PatternDetector
from .query import ONNXQuery

logger = logging.getLogger("onnx_toolkit")


class ONNXParser:
    """
    Load an ONNX model and expose it for querying and pattern detection.

    Parameters
    ----------
    onnx_path : str  Path to the ``.onnx`` file.

    Example
    -------
    >>> parser = ONNXParser("model.onnx")
    >>> convs = parser.find().find_by_op_type("Conv").has_params()
    >>> print(convs)
    """

    def __init__(self, onnx_path: str) -> None:
        logger.info("Loading ONNX model from %r", onnx_path)
        self.model: ModelProto = onnx.load(onnx_path)
        self.nodes: List[NodeProto] = list(self.model.graph.node)
        self.tensor_map: TensorMap = {
            t.name: numpy_helper.to_array(t)
            for t in self.model.graph.initializer
        }
        self.graph_inputs: Set[str] = {i.name for i in self.model.graph.input}
        self.graph_outputs: Set[str] = {o.name for o in self.model.graph.output}
        logger.info(
            "Model loaded: %d nodes, %d tensors, %d graph input(s), "
            "%d graph output(s)",
            len(self.nodes),
            len(self.tensor_map),
            len(self.graph_inputs),
            len(self.graph_outputs),
        )
        logger.debug("Graph inputs : %s", self.graph_inputs)
        logger.debug("Graph outputs: %s", self.graph_outputs)

    # ------------------------------------------------------------------

    def find(self) -> ONNXQuery:
        """Return an ONNXQuery over **all** nodes in the graph."""
        logger.debug("find(): returning ONNXQuery over all %d nodes", len(self.nodes))
        return ONNXQuery(
            list(self.nodes),
            self.tensor_map,
            self.nodes,
            self.graph_inputs,
            self.graph_outputs,
        )

    def pattern_detect(
        self,
        pattern: Pattern,
        start_node: Optional[Union[str, NodeProto]] = None,
        end_node: Optional[Union[str, NodeProto]] = None,
    ) -> Optional[NodeProto]:
        """
        Convenience shortcut: create a PatternDetector bound to this model
        and immediately call ``match()``.

        Parameters
        ----------
        pattern    : Pattern   The pattern to check.
        start_node : optional  Seed node (name or NodeProto).
        end_node   : optional  Exclusive boundary node (name or NodeProto).
                               See :class:`PatternDetector` for the full
                               end-node contract.

        Returns
        -------
        NodeProto
            The terminating node of the match (end_node if reached, otherwise
            start_node), or ``None`` if the pattern did not match.
        """
        logger.debug(
            "pattern_detect(start_node=%r, end_node=%r)",
            start_node if isinstance(start_node, str)
            else getattr(start_node, "name", start_node),
            end_node if isinstance(end_node, str)
            else getattr(end_node, "name", end_node),
        )
        detector = PatternDetector(
            self.model, start_node=start_node, end_node=end_node
        )
        result = detector.match(pattern)
        logger.debug(
            "pattern_detect() → %s",
            result.name if result is not None else None,
        )
        return result

    def summary(self) -> str:
        """Return a human-readable summary of the model."""
        op_counts = Counter(n.op_type for n in self.nodes)
        lines = [
            "ONNX model summary",
            f"  Nodes       : {len(self.nodes)}",
            f"  Tensors     : {len(self.tensor_map)}",
            f"  Graph inputs: {len(self.graph_inputs)}",
            f"  Graph outputs:{len(self.graph_outputs)}",
            f"  Op types ({len(op_counts)}):",
        ]
        for op, cnt in op_counts.most_common():
            lines.append(f"    {op:<24} {cnt:>5}")
        summary_str = "\n".join(lines)
        logger.debug("summary():\n%s", summary_str)
        return summary_str
