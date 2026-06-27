"""TFLite specific utilities."""

from __future__ import annotations

import logging
from typing import Any

from neuron_toolkit._utils import ShapeInfo

log = logging.getLogger("neuron_toolkit.backends.tflite")

# Map TFLite TensorType (int) -> numpy dtype string
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


# TODO: Extract attributes for additional common operators
# (e.g. FULLY_CONNECTED, CONCATENATION, ADD, SUB, etc.)
def _get_tflite_attr(  # noqa: PLR0912, PLR0915
    op: Any, op_type: str
) -> dict[str, object]:
    """Try to extract attributes from a TFLite operator.

    This is challenging because TFLite uses specific flatbuffer tables for
    each op's options.
    """
    options = op.BuiltinOptions()
    if options is None:
        return {}

    attrs: dict[str, Any] = {}

    try:
        if op_type == "CONV_2D":
            from tflite.Conv2DOptions import Conv2DOptions  # noqa: PLC0415

            opt = Conv2DOptions()
            opt.Init(options.Bytes, options.Pos)
            attrs["stride_h"] = opt.StrideH()
            attrs["stride_w"] = opt.StrideW()
            attrs["padding"] = opt.Padding()
            attrs["fused_activation_function"] = opt.FusedActivationFunction()
            attrs["dilation_h_factor"] = opt.DilationHFactor()
            attrs["dilation_w_factor"] = opt.DilationWFactor()
        elif op_type in {"POOL_2D", "AVERAGE_POOL_2D", "MAX_POOL_2D"}:
            from tflite.Pool2DOptions import Pool2DOptions  # noqa: PLC0415

            opt = Pool2DOptions()
            opt.Init(options.Bytes, options.Pos)
            attrs["stride_h"] = opt.StrideH()
            attrs["stride_w"] = opt.StrideW()
            attrs["padding"] = opt.Padding()
            attrs["filter_height"] = opt.FilterHeight()
            attrs["filter_width"] = opt.FilterWidth()
            attrs["fused_activation_function"] = opt.FusedActivationFunction()
        elif op_type == "RESHAPE":
            from tflite.ReshapeOptions import ReshapeOptions  # noqa: PLC0415

            opt = ReshapeOptions()
            opt.Init(options.Bytes, options.Pos)
        elif op_type == "SOFTMAX":
            from tflite.SoftmaxOptions import SoftmaxOptions  # noqa: PLC0415

            opt = SoftmaxOptions()
            opt.Init(options.Bytes, options.Pos)
            attrs["beta"] = opt.Beta()
        elif op_type == "FULLY_CONNECTED":
            from tflite.FullyConnectedOptions import (  # noqa: PLC0415
                FullyConnectedOptions,
            )

            opt = FullyConnectedOptions()
            opt.Init(options.Bytes, options.Pos)
            attrs["fused_activation_function"] = opt.FusedActivationFunction()
            attrs["keep_num_dims"] = opt.KeepNumDims()
        elif op_type == "CONCATENATION":
            from tflite.ConcatenationOptions import (  # noqa: PLC0415
                ConcatenationOptions,
            )

            opt = ConcatenationOptions()
            opt.Init(options.Bytes, options.Pos)
            attrs["axis"] = opt.Axis()
            attrs["fused_activation_function"] = opt.FusedActivationFunction()
        elif op_type == "ADD":
            from tflite.AddOptions import AddOptions  # noqa: PLC0415

            opt = AddOptions()
            opt.Init(options.Bytes, options.Pos)
            attrs["fused_activation_function"] = opt.FusedActivationFunction()
        elif op_type == "SUB":
            from tflite.SubOptions import SubOptions  # noqa: PLC0415

            opt = SubOptions()
            opt.Init(options.Bytes, options.Pos)
            attrs["fused_activation_function"] = opt.FusedActivationFunction()
        elif op_type == "MUL":
            from tflite.MulOptions import MulOptions  # noqa: PLC0415

            opt = MulOptions()
            opt.Init(options.Bytes, options.Pos)
            attrs["fused_activation_function"] = opt.FusedActivationFunction()
        elif op_type == "DIV":
            from tflite.DivOptions import DivOptions  # noqa: PLC0415

            opt = DivOptions()
            opt.Init(options.Bytes, options.Pos)
            attrs["fused_activation_function"] = opt.FusedActivationFunction()
        elif op_type == "SQUEEZE":
            from tflite.SqueezeOptions import SqueezeOptions  # noqa: PLC0415

            opt = SqueezeOptions()
            opt.Init(options.Bytes, options.Pos)
            dims = [
                opt.SqueezeDims(j) for j in range(opt.SqueezeDimsLength())
            ]
            attrs["squeeze_dims"] = dims
    except Exception as exc:  # noqa: BLE001
        log.debug("Failed to extract attributes for %s: %s", op_type, exc)

    return attrs


def _build_shape_info(_model: object, subgraph: Any) -> ShapeInfo:
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
