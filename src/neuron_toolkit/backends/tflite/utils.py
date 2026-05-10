"""TFLite specific utilities."""

from __future__ import annotations

import logging
from typing import Any

from neuron_toolkit._utils import ShapeInfo

log = logging.getLogger("neuron_toolkit.backends.tflite")

# Map TFLite TensorType (int) -> numpy dtype string
# Based on tflite.TensorType
_TFLITE_DTYPE_TO_NP: dict[int, str] = {
    0: "float32",
    1: "float16",
    2: "int32",
    3: "uint8",
    4: "int64",
    5: "object",  # String
    6: "bool",
    7: "int16",
    8: "complex64",
    9: "int8",
    10: "float64",
    11: "complex128",
    12: "uint64",
    15: "uint32",
}


def _get_tflite_attr(op: Any, op_type: str) -> dict[str, Any]:
    """Try to extract attributes from a TFLite operator.

    This is challenging because TFLite uses specific flatbuffer tables for each op's options.
    """

    options = op.BuiltinOptions()
    if options is None:
        return {}

    attrs = {}

    # This is a very simplified version. Real implementation would need to handle
    # many more op types and their corresponding Options classes.
    try:
        if op_type == "CONV_2D":
            from tflite.Conv2DOptions import Conv2DOptions

            opt = Conv2DOptions()
            opt.Init(options.Bytes, options.Pos)
            attrs["stride_h"] = opt.StrideH()
            attrs["stride_w"] = opt.StrideW()
            attrs["padding"] = opt.Padding()
            attrs["fused_activation_function"] = opt.FusedActivationFunction()
            attrs["dilation_h_factor"] = opt.DilationHFactor()
            attrs["dilation_w_factor"] = opt.DilationWFactor()
        elif (
            op_type == "POOL_2D"
            or op_type == "AVERAGE_POOL_2D"
            or op_type == "MAX_POOL_2D"
        ):
            from tflite.Pool2DOptions import Pool2DOptions

            opt = Pool2DOptions()
            opt.Init(options.Bytes, options.Pos)
            attrs["stride_h"] = opt.StrideH()
            attrs["stride_w"] = opt.StrideW()
            attrs["padding"] = opt.Padding()
            attrs["filter_height"] = opt.FilterHeight()
            attrs["filter_width"] = opt.FilterWidth()
            attrs["fused_activation_function"] = opt.FusedActivationFunction()
        elif op_type == "RESHAPE":
            from tflite.ReshapeOptions import ReshapeOptions

            opt = ReshapeOptions()
            opt.Init(options.Bytes, options.Pos)
            # Reshape options might have 'new_shape' but it's often in a separate tensor
            pass
        elif op_type == "SOFTMAX":
            from tflite.SoftmaxOptions import SoftmaxOptions

            opt = SoftmaxOptions()
            opt.Init(options.Bytes, options.Pos)
            attrs["beta"] = opt.Beta()
        # Add more as needed...
    except Exception as exc:
        log.debug("Failed to extract attributes for %s: %s", op_type, exc)

    return attrs


def _build_shape_info(model: Any, subgraph: Any) -> ShapeInfo:
    """Build a map from every tensor name to (rank, dtype)."""
    info: ShapeInfo = {}

    for i in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(i)
        name = tensor.Name().decode("utf-8")
        shape = [tensor.Shape(j) for j in range(tensor.ShapeLength())]
        dtype_code = tensor.Type()
        dtype = _TFLITE_DTYPE_TO_NP.get(dtype_code, "unknown")
        info[name] = (len(shape), dtype)

    return info
