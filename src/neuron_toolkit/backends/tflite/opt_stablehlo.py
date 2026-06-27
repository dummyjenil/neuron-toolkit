"""TFLite options parser for stablehlo operations."""
# ruff: noqa: PLC0415, N802

from __future__ import annotations

from typing import Any


def parse_DilateOptions(_options: Any) -> dict[str, Any]:
    """Parse DilateOptions attributes."""
    return {}


def parse_ReduceWindowOptions(_options: Any) -> dict[str, Any]:
    """Parse ReduceWindowOptions attributes."""
    from tflite.ReduceWindowOptions import ReduceWindowOptions

    opt = ReduceWindowOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["reduce_function"] = opt.ReduceFunction()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StableHLOCompositeOptions(_options: Any) -> dict[str, Any]:
    """Parse StableHLOCompositeOptions attributes."""
    from tflite.StableHLOCompositeOptions import StableHLOCompositeOptions

    opt = StableHLOCompositeOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["composite_attributes_format"] = opt.CompositeAttributesFormat()
    attrs["decomposition_subgraph_index"] = opt.DecompositionSubgraphIndex()
    attrs["name"] = opt.Name()
    attrs["version"] = opt.Version()
    attrs["composite_attributes"] = opt.CompositeAttributesAsNumpy()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloBroadcastInDimOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloBroadcastInDimOptions attributes."""
    from tflite.StablehloBroadcastInDimOptions import StablehloBroadcastInDimOptions

    opt = StablehloBroadcastInDimOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["broadcast_dimensions"] = opt.BroadcastDimensionsAsNumpy()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloCompareOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloCompareOptions attributes."""
    from tflite.StablehloCompareOptions import StablehloCompareOptions

    opt = StablehloCompareOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["compare_type"] = opt.CompareType()
    attrs["comparison_direction"] = opt.ComparisonDirection()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloConcatenateOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloConcatenateOptions attributes."""
    from tflite.StablehloConcatenateOptions import StablehloConcatenateOptions

    opt = StablehloConcatenateOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["dimension"] = opt.Dimension()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloConvolutionOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloConvolutionOptions attributes."""
    from tflite.StablehloConvolutionOptions import StablehloConvolutionOptions

    opt = StablehloConvolutionOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["batch_group_count"] = opt.BatchGroupCount()
    attrs["feature_group_count"] = opt.FeatureGroupCount()
    attrs["input_batch_dimension"] = opt.InputBatchDimension()
    attrs["input_feature_dimension"] = opt.InputFeatureDimension()
    attrs["kernel_input_feature_dimension"] = opt.KernelInputFeatureDimension()
    attrs["kernel_output_feature_dimension"] = opt.KernelOutputFeatureDimension()
    attrs["output_batch_dimension"] = opt.OutputBatchDimension()
    attrs["output_feature_dimension"] = opt.OutputFeatureDimension()
    attrs["input_spatial_dimensions"] = opt.InputSpatialDimensionsAsNumpy()
    attrs["kernel_spatial_dimensions"] = opt.KernelSpatialDimensionsAsNumpy()
    attrs["lhs_dilation"] = opt.LhsDilationAsNumpy()
    attrs["output_spatial_dimensions"] = opt.OutputSpatialDimensionsAsNumpy()
    attrs["padding"] = opt.PaddingAsNumpy()
    attrs["precision_config"] = opt.PrecisionConfigAsNumpy()
    attrs["rhs_dilation"] = opt.RhsDilationAsNumpy()
    attrs["window_reversal"] = opt.WindowReversalAsNumpy()
    attrs["window_strides"] = opt.WindowStridesAsNumpy()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloCustomCallOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloCustomCallOptions attributes."""
    from tflite.StablehloCustomCallOptions import StablehloCustomCallOptions

    opt = StablehloCustomCallOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["api_version"] = opt.ApiVersion()
    attrs["backend_config"] = opt.BackendConfig()
    attrs["call_target_name"] = opt.CallTargetName()
    attrs["has_side_effect"] = opt.HasSideEffect()
    attrs["called_computations"] = opt.CalledComputationsAsNumpy()
    attrs["custom_attributes"] = opt.CustomAttributesAsNumpy()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloDotGeneralOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloDotGeneralOptions attributes."""
    from tflite.StablehloDotGeneralOptions import StablehloDotGeneralOptions

    opt = StablehloDotGeneralOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["lhs_batching_dimensions"] = opt.LhsBatchingDimensionsAsNumpy()
    attrs["lhs_contracting_dimensions"] = opt.LhsContractingDimensionsAsNumpy()
    attrs["precision_config"] = opt.PrecisionConfigAsNumpy()
    attrs["rhs_batching_dimensions"] = opt.RhsBatchingDimensionsAsNumpy()
    attrs["rhs_contracting_dimensions"] = opt.RhsContractingDimensionsAsNumpy()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloDynamicSliceOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloDynamicSliceOptions attributes."""
    from tflite.StablehloDynamicSliceOptions import StablehloDynamicSliceOptions

    opt = StablehloDynamicSliceOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["slice_sizes"] = opt.SliceSizesAsNumpy()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloGatherOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloGatherOptions attributes."""
    from tflite.StablehloGatherOptions import StablehloGatherOptions

    opt = StablehloGatherOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["index_vector_dim"] = opt.IndexVectorDim()
    attrs["indices_are_sorted"] = opt.IndicesAreSorted()
    attrs["collapsed_slice_dims"] = opt.CollapsedSliceDimsAsNumpy()
    attrs["offset_dims"] = opt.OffsetDimsAsNumpy()
    attrs["slice_sizes"] = opt.SliceSizesAsNumpy()
    attrs["start_index_map"] = opt.StartIndexMapAsNumpy()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloIotaOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloIotaOptions attributes."""
    from tflite.StablehloIotaOptions import StablehloIotaOptions

    opt = StablehloIotaOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["iota_dimension"] = opt.IotaDimension()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloPadOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloPadOptions attributes."""
    from tflite.StablehloPadOptions import StablehloPadOptions

    opt = StablehloPadOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["edge_padding_high"] = opt.EdgePaddingHighAsNumpy()
    attrs["edge_padding_low"] = opt.EdgePaddingLowAsNumpy()
    attrs["interior_padding"] = opt.InteriorPaddingAsNumpy()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloReduceOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloReduceOptions attributes."""
    from tflite.StablehloReduceOptions import StablehloReduceOptions

    opt = StablehloReduceOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["body_subgraph_index"] = opt.BodySubgraphIndex()
    attrs["dimensions"] = opt.DimensionsAsNumpy()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloReduceWindowOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloReduceWindowOptions attributes."""
    from tflite.StablehloReduceWindowOptions import StablehloReduceWindowOptions

    opt = StablehloReduceWindowOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["body_subgraph_index"] = opt.BodySubgraphIndex()
    attrs["base_dilations"] = opt.BaseDilationsAsNumpy()
    attrs["padding"] = opt.PaddingAsNumpy()
    attrs["window_dilations"] = opt.WindowDilationsAsNumpy()
    attrs["window_dimensions"] = opt.WindowDimensionsAsNumpy()
    attrs["window_strides"] = opt.WindowStridesAsNumpy()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloRngBitGeneratorOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloRngBitGeneratorOptions attributes."""
    from tflite.StablehloRngBitGeneratorOptions import StablehloRngBitGeneratorOptions

    opt = StablehloRngBitGeneratorOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["algorithm"] = opt.Algorithm()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloScatterOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloScatterOptions attributes."""
    from tflite.StablehloScatterOptions import StablehloScatterOptions

    opt = StablehloScatterOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["index_vector_dim"] = opt.IndexVectorDim()
    attrs["indices_are_sorted"] = opt.IndicesAreSorted()
    attrs["unique_indices"] = opt.UniqueIndices()
    attrs["update_computation_subgraph_index"] = opt.UpdateComputationSubgraphIndex()
    attrs["inserted_window_dims"] = opt.InsertedWindowDimsAsNumpy()
    attrs["scatter_dims_to_operand_dims"] = opt.ScatterDimsToOperandDimsAsNumpy()
    attrs["update_window_dims"] = opt.UpdateWindowDimsAsNumpy()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloShiftLeftOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloShiftLeftOptions attributes."""
    return {}


def parse_StablehloSliceOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloSliceOptions attributes."""
    from tflite.StablehloSliceOptions import StablehloSliceOptions

    opt = StablehloSliceOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["limit_indices"] = opt.LimitIndicesAsNumpy()
    attrs["start_indices"] = opt.StartIndicesAsNumpy()
    attrs["strides"] = opt.StridesAsNumpy()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloSortOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloSortOptions attributes."""
    from tflite.StablehloSortOptions import StablehloSortOptions

    opt = StablehloSortOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["comparator_subgraph_index"] = opt.ComparatorSubgraphIndex()
    attrs["dimension"] = opt.Dimension()
    attrs["is_stable"] = opt.IsStable()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloTransposeOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloTransposeOptions attributes."""
    from tflite.StablehloTransposeOptions import StablehloTransposeOptions

    opt = StablehloTransposeOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["permutation"] = opt.PermutationAsNumpy()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_StablehloWhileOptions(_options: Any) -> dict[str, Any]:
    """Parse StablehloWhileOptions attributes."""
    from tflite.StablehloWhileOptions import StablehloWhileOptions

    opt = StablehloWhileOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["body_subgraph_index"] = opt.BodySubgraphIndex()
    attrs["cond_subgraph_index"] = opt.CondSubgraphIndex()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs
