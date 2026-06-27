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


_OPT_TYPE_TO_CLASS: dict[int, str] = {}
_OPT2_TYPE_TO_CLASS: dict[int, str] = {}


def _init_opt_mappings() -> None:
    """Initialize option type to class name mappings."""
    if _OPT_TYPE_TO_CLASS:
        return
    try:
        import tflite  # noqa: PLC0415

        if hasattr(tflite, "BuiltinOptions"):
            for name in dir(tflite.BuiltinOptions):
                if not name.startswith("_") and name != "NONE":
                    val = getattr(tflite.BuiltinOptions, name)
                    _OPT_TYPE_TO_CLASS[val] = name
        if hasattr(tflite, "BuiltinOptions2"):
            for name in dir(tflite.BuiltinOptions2):
                if not name.startswith("_") and name != "NONE":
                    val = getattr(tflite.BuiltinOptions2, name)
                    _OPT2_TYPE_TO_CLASS[val] = name
    except ImportError:
        pass


# Map options class name to category suffix
_OPT_CLASS_TO_CATEGORY: dict[str, str] = {
    "SoftmaxOptions": "activation",
    "HardSwishOptions": "activation",
    "LeakyReluOptions": "activation",
    "GeluOptions": "activation",
    "AddOptions": "arithmetic",
    "SubOptions": "arithmetic",
    "MulOptions": "arithmetic",
    "DivOptions": "arithmetic",
    "NegOptions": "arithmetic",
    "AbsOptions": "arithmetic",
    "SquareOptions": "arithmetic",
    "SquaredDifferenceOptions": "arithmetic",
    "PowOptions": "arithmetic",
    "ExpOptions": "arithmetic",
    "LogSoftmaxOptions": "arithmetic",
    "FloorDivOptions": "arithmetic",
    "FloorModOptions": "arithmetic",
    "AddNOptions": "arithmetic",
    "CosOptions": "arithmetic",
    "ATan2Options": "arithmetic",
    "SignOptions": "arithmetic",
    "MaximumMinimumOptions": "arithmetic",
    "BitwiseXorOptions": "arithmetic",
    "RightShiftOptions": "arithmetic",
    "ReshapeOptions": "tensor",
    "SqueezeOptions": "tensor",
    "TransposeOptions": "tensor",
    "SliceOptions": "tensor",
    "StridedSliceOptions": "tensor",
    "ConcatenationOptions": "tensor",
    "SplitOptions": "tensor",
    "SplitVOptions": "tensor",
    "TileOptions": "tensor",
    "PackOptions": "tensor",
    "UnpackOptions": "tensor",
    "ExpandDimsOptions": "tensor",
    "BroadcastToOptions": "tensor",
    "ReverseV2Options": "tensor",
    "ReverseSequenceOptions": "tensor",
    "GatherOptions": "tensor",
    "GatherNdOptions": "tensor",
    "ScatterNdOptions": "tensor",
    "OneHotOptions": "tensor",
    "ShapeOptions": "tensor",
    "UniqueOptions": "tensor",
    "RankOptions": "tensor",
    "CastOptions": "tensor",
    "BitcastOptions": "tensor",
    "FillOptions": "tensor",
    "ZerosLikeOptions": "tensor",
    "Conv2DOptions": "nn",
    "Conv3DOptions": "nn",
    "DepthwiseConv2DOptions": "nn",
    "TransposeConvOptions": "nn",
    "FullyConnectedOptions": "nn",
    "BatchMatMulOptions": "nn",
    "Pool2DOptions": "nn",
    "L2NormOptions": "nn",
    "LocalResponseNormalizationOptions": "nn",
    "SVDFOptions": "nn",
    "ReducerOptions": "reduction",
    "ArgMaxOptions": "reduction",
    "ArgMinOptions": "reduction",
    "EqualOptions": "logical",
    "NotEqualOptions": "logical",
    "LessOptions": "logical",
    "LessEqualOptions": "logical",
    "GreaterOptions": "logical",
    "GreaterEqualOptions": "logical",
    "LogicalAndOptions": "logical",
    "LogicalOrOptions": "logical",
    "LogicalNotOptions": "logical",
    "SelectOptions": "logical",
    "SelectV2Options": "logical",
    "WhereOptions": "logical",
    "IfOptions": "control_flow",
    "WhileOptions": "control_flow",
    "CallOptions": "control_flow",
    "CallOnceOptions": "control_flow",
    "DilateOptions": "stablehlo",
    "ReduceWindowOptions": "stablehlo",
    "StableHLOCompositeOptions": "stablehlo",
    "StablehloBroadcastInDimOptions": "stablehlo",
    "StablehloCompareOptions": "stablehlo",
    "StablehloConcatenateOptions": "stablehlo",
    "StablehloConvolutionOptions": "stablehlo",
    "StablehloCustomCallOptions": "stablehlo",
    "StablehloDotGeneralOptions": "stablehlo",
    "StablehloDynamicSliceOptions": "stablehlo",
    "StablehloGatherOptions": "stablehlo",
    "StablehloIotaOptions": "stablehlo",
    "StablehloPadOptions": "stablehlo",
    "StablehloReduceOptions": "stablehlo",
    "StablehloReduceWindowOptions": "stablehlo",
    "StablehloRngBitGeneratorOptions": "stablehlo",
    "StablehloScatterOptions": "stablehlo",
    "StablehloShiftLeftOptions": "stablehlo",
    "StablehloSliceOptions": "stablehlo",
    "StablehloSortOptions": "stablehlo",
    "StablehloTransposeOptions": "stablehlo",
    "StablehloWhileOptions": "stablehlo",
    "AssignVariableOptions": "misc",
    "BatchToSpaceNDOptions": "misc",
    "BidirectionalSequenceLSTMOptions": "misc",
    "BidirectionalSequenceRNNOptions": "misc",
    "BucketizeOptions": "misc",
    "ConcatEmbeddingsOptions": "misc",
    "CumsumOptions": "misc",
    "DensifyOptions": "misc",
    "DepthToSpaceOptions": "misc",
    "DequantizeOptions": "misc",
    "DynamicUpdateSliceOptions": "misc",
    "EmbeddingLookupSparseOptions": "misc",
    "FakeQuantOptions": "misc",
    "HashtableFindOptions": "misc",
    "HashtableImportOptions": "misc",
    "HashtableOptions": "misc",
    "HashtableSizeOptions": "misc",
    "LSHProjectionOptions": "misc",
    "LSTMOptions": "misc",
    "MatrixDiagOptions": "misc",
    "MatrixSetDiagOptions": "misc",
    "MirrorPadOptions": "misc",
    "NonMaxSuppressionV4Options": "misc",
    "NonMaxSuppressionV5Options": "misc",
    "PadOptions": "misc",
    "PadV2Options": "misc",
    "QuantizeOptions": "misc",
    "RNNOptions": "misc",
    "RandomOptions": "misc",
    "RangeOptions": "misc",
    "ReadVariableOptions": "misc",
    "ResizeBilinearOptions": "misc",
    "ResizeNearestNeighborOptions": "misc",
    "Rfft2dOptions": "misc",
    "SegmentSumOptions": "misc",
    "SequenceRNNOptions": "misc",
    "SkipGramOptions": "misc",
    "SpaceToBatchNDOptions": "misc",
    "SpaceToDepthOptions": "misc",
    "SparseToDenseOptions": "misc",
    "TopKV2Options": "misc",
    "UnidirectionalSequenceLSTMOptions": "misc",
    "UnsortedSegmentMaxOptions": "misc",
    "UnsortedSegmentMinOptions": "misc",
    "UnsortedSegmentProdOptions": "misc",
    "UnsortedSegmentSumOptions": "misc",
    "VarHandleOptions": "misc",
}


def _get_tflite_attr(op: Any, op_type: str) -> dict[str, object]:
    """Try to extract attributes from a TFLite operator.

    This is challenging because TFLite uses specific flatbuffer tables for
    each op's options.
    """
    options = op.BuiltinOptions()
    if options is None:
        return {}

    _init_opt_mappings()

    builtin_opt_type = op.BuiltinOptionsType()
    builtin_opt2_type = op.BuiltinOptions2Type()

    opt_class_name = None
    if builtin_opt_type != 0:
        opt_class_name = _OPT_TYPE_TO_CLASS.get(builtin_opt_type)
    elif builtin_opt2_type != 0:
        opt_class_name = _OPT2_TYPE_TO_CLASS.get(builtin_opt2_type)

    if not opt_class_name:
        return {}

    category = _OPT_CLASS_TO_CATEGORY.get(opt_class_name)
    if not category:
        return {}

    try:
        # Dynamically import the category module and the specific parse function
        module_name = f"neuron_toolkit.backends.tflite.opt_{category}"
        module = __import__(module_name, fromlist=[f"parse_{opt_class_name}"])
        parse_func = getattr(module, f"parse_{opt_class_name}")
        return parse_func(options)
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "Failed to extract attributes for %s (class: %s, category: %s): %s",
            op_type,
            opt_class_name,
            category,
            exc,
        )

    return {}


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
