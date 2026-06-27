"""ONNX specific utilities."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
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

_ATTR_EXTRACTORS: dict[int, Callable[[AttributeProto], object]] = {
    AttributeProto.FLOAT: lambda a: a.f,
    AttributeProto.INT: lambda a: a.i,
    AttributeProto.STRING: lambda a: a.s.decode("utf-8"),
    AttributeProto.TENSOR: lambda a: to_array(a.t),
    AttributeProto.FLOATS: lambda a: list(a.floats),
    AttributeProto.INTS: lambda a: list(a.ints),
    AttributeProto.STRINGS: lambda a: [s.decode("utf-8") for s in a.strings],
    AttributeProto.TENSORS: lambda a: [to_array(ten) for ten in a.tensors],
}


def _attr_value(attr: AttributeProto) -> object | None:
    """Extract a typed Python value from an ONNX AttributeProto."""
    extractor = _ATTR_EXTRACTORS.get(attr.type)
    if extractor:
        return extractor(attr)

    log.debug(
        "_attr_value: unrecognised attribute type %s for %r",
        attr.type,
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

    def _extract_vi(vi_list: Sequence[onnx.ValueInfoProto]) -> None:
        for vi in vi_list:
            if not vi.type.HasField("tensor_type"):
                continue
            t = vi.type.tensor_type
            rank = len(t.shape.dim) if t.HasField("shape") else None
            dtype = _ONNX_DTYPE_TO_NP.get(t.elem_type)
            info[vi.name] = (rank, dtype)

    # Include inputs, value_info, and outputs
    _extract_vi(list(inferred_model.graph.input))
    _extract_vi(list(inferred_model.graph.value_info))
    _extract_vi(list(inferred_model.graph.output))

    return info
