"""ONNX specific utilities."""

from __future__ import annotations

import logging
from typing import Any

import onnx
from onnx import AttributeProto, TensorProto
from onnx.numpy_helper import to_array
from onnx.onnx_pb import NodeProto

from neuron_toolkit._utils import ShapeInfo

log = logging.getLogger("neuron_toolkit.backends.onnx")

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
    if t == AttributeProto.FLOAT:
        return attr.f
    if t == AttributeProto.INT:
        return attr.i
    if t == AttributeProto.STRING:
        return attr.s.decode("utf-8")
    if t == AttributeProto.TENSOR:
        return to_array(attr.t)
    if t == AttributeProto.FLOATS:
        return list(attr.floats)
    if t == AttributeProto.INTS:
        return list(attr.ints)
    if t == AttributeProto.STRINGS:
        return [s.decode("utf-8") for s in attr.strings]
    if t == AttributeProto.TENSORS:
        return [to_array(ten) for ten in attr.tensors]

    log.debug(
        "_attr_value: unrecognised attribute type %s for %r",
        t,
        getattr(attr, "name", "?"),
    )
    return None


def _node_attrs(node: NodeProto) -> dict[str, Any]:
    """Return all attributes of *node* as a plain dict."""
    return {a.name: _attr_value(a) for a in node.attribute}


def _build_shape_info(inferred_model: onnx.ModelProto) -> ShapeInfo:
    """Build a map from every value name to (rank, dtype).

    Uses the type/shape annotations produced by onnx.shape_inference.infer_shapes.
    """
    info: ShapeInfo = {}

    def _extract_vi(vi_list):
        for vi in vi_list:
            if not vi.type.HasField("tensor_type"):
                continue
            t = vi.type.tensor_type
            rank = len(t.shape.dim) if t.HasField("shape") else None
            dtype = _ONNX_DTYPE_TO_NP.get(t.elem_type)
            info[vi.name] = (rank, dtype)

    # Include inputs, value_info, and outputs
    _extract_vi(inferred_model.graph.input)
    _extract_vi(inferred_model.graph.value_info)
    _extract_vi(inferred_model.graph.output)

    return info
