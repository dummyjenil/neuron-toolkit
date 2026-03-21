"""
onnx_toolkit._utils
====================
Internal helpers shared across the package.

  _attr_value  – extract a typed Python value from an ONNX AttributeProto.
  _GraphShim   – lightweight stand-in for ModelProto used by Pattern.detect
                 when called from ONNXQuery.matches().
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import onnx
import numpy as np
from onnx import numpy_helper
from onnx.onnx_pb import NodeProto

from ._types import TensorMap

logger = logging.getLogger("onnx_toolkit")


# ---------------------------------------------------------------------------
# Attribute extraction
# ---------------------------------------------------------------------------

def _attr_value(attr: Any) -> Any:
    """Extract a typed Python value from an ONNX AttributeProto."""
    t = attr.type
    if t == onnx.AttributeProto.FLOAT:
        return attr.f
    if t == onnx.AttributeProto.INT:
        return attr.i
    if t == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8")
    if t == onnx.AttributeProto.TENSOR:
        return numpy_helper.to_array(attr.t)
    if t == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    if t == onnx.AttributeProto.INTS:
        return list(attr.ints)
    if t == onnx.AttributeProto.STRINGS:
        return [s.decode("utf-8") for s in attr.strings]
    logger.debug(
        "_attr_value: unrecognised attribute type %d for %r", t, attr.name
    )
    return None


# ---------------------------------------------------------------------------
# Graph shim
# ---------------------------------------------------------------------------

class _GraphShim:
    """
    Minimal ModelProto-like object used to pass graph data from
    ONNXQuery into PatternDetector without requiring a full ModelProto.
    """

    def __init__(self, nodes: List[NodeProto], tensor_map: TensorMap) -> None:
        self.nodes = nodes
        self.tensor_map = tensor_map
        logger.debug(
            "_GraphShim created: %d nodes, %d tensors",
            len(nodes), len(tensor_map),
        )
