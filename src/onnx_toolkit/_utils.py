"""onnx_toolkit._utils — internal helpers."""

from __future__ import annotations

import logging
from typing import Any, Optional

import onnx
from onnx import AttributeProto, TensorProto
from onnx.numpy_helper import to_array
from onnx.onnx_pb import NodeProto

from ._types import TensorMap

log = logging.getLogger("onnx_toolkit")

# Map ONNX dtype int → numpy dtype string (covers all standard types)
_ONNX_DTYPE_TO_NP: dict[int, str] = {
    TensorProto.FLOAT: "float32",
    TensorProto.DOUBLE: "float64",
    TensorProto.INT8: "int8",
    TensorProto.INT16: "int16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.UINT8: "uint8",
    TensorProto.UINT16: "uint16",
    TensorProto.UINT32: "uint32",
    TensorProto.UINT64: "uint64",
    TensorProto.BOOL: "bool",
    TensorProto.FLOAT16: "float16",
    TensorProto.STRING: "object",
}

def _attr_value(attr: AttributeProto) -> object | None:
    """Extract a typed Python value from an ONNX AttributeProto."""
    t = attr.type

    scalar_extractors: dict[int, object] = {
        AttributeProto.FLOAT: attr.f,
        AttributeProto.INT: attr.i,
        AttributeProto.STRING: attr.s.decode("utf-8"),
        AttributeProto.TENSOR: to_array(attr.t),
        AttributeProto.FLOATS: list(attr.floats),
        AttributeProto.INTS: list(attr.ints),
        AttributeProto.STRINGS: [
            s.decode("utf-8") for s in attr.strings
        ],
    }

    value = scalar_extractors.get(t)

    if value is None:
        log.debug(
            "_attr_value: unrecognised attribute type %s for %r",
            t,
            getattr(attr, "name", "?"),
        )

    return value

def _node_attrs(node: NodeProto) -> dict[str, Any]:
    """Return all attributes of *node* as a plain dict."""
    return {a.name: _attr_value(a) for a in node.attribute}


# ---------------------------------------------------------------------------
# Shape-info helpers (populated from onnx.shape_inference)
# ---------------------------------------------------------------------------

# ShapeInfo: output_name → (rank | None, dtype_str | None)
ShapeInfo = dict[str, tuple[Optional[int], Optional[str]]]


def _build_shape_info(inferred_model: onnx.ModelProto) -> ShapeInfo:
    """Build a map from every value name to (rank, dtype).

    Uses the type/shape annotations produced by onnx.shape_inference.infer_shapes.
    """
    info: ShapeInfo = {}
    for vi in inferred_model.graph.value_info:
        t = vi.type.tensor_type
        rank = len(t.shape.dim) if t.HasField("shape") else None
        dtype = _ONNX_DTYPE_TO_NP.get(t.elem_type)
        info[vi.name] = (rank, dtype)
    # Also include graph outputs
    for vi in inferred_model.graph.output:
        t = vi.type.tensor_type
        rank = len(t.shape.dim) if t.HasField("shape") else None
        dtype = _ONNX_DTYPE_TO_NP.get(t.elem_type)
        info[vi.name] = (rank, dtype)
    return info


# ---------------------------------------------------------------------------
# Graph shim
# ---------------------------------------------------------------------------


class _GraphShim:
    """Lightweight stand-in for ModelProto used inside PatternDetector."""

    def __init__(
        self,
        nodes: list[NodeProto],
        tensor_map: TensorMap,
        shape_info: ShapeInfo | None = None,
    ) -> None:
        self.nodes = nodes
        self.tensor_map = tensor_map
        self.shape_info: ShapeInfo = shape_info or {}
