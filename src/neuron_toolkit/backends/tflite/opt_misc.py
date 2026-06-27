"""TFLite options parser for misc operations."""
# ruff: noqa: PLC0415, N802

from __future__ import annotations

from typing import Any


def parse_AssignVariableOptions(_options: Any) -> dict[str, Any]:
    """Parse AssignVariableOptions attributes."""
    return {}


def parse_BatchToSpaceNDOptions(_options: Any) -> dict[str, Any]:
    """Parse BatchToSpaceNDOptions attributes."""
    return {}


def parse_BidirectionalSequenceLSTMOptions(_options: Any) -> dict[str, Any]:
    """Parse BidirectionalSequenceLSTMOptions attributes."""
    from tflite.BidirectionalSequenceLSTMOptions import BidirectionalSequenceLSTMOptions

    opt = BidirectionalSequenceLSTMOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["asymmetric_quantize_inputs"] = opt.AsymmetricQuantizeInputs()
    attrs["cell_clip"] = opt.CellClip()
    attrs["fused_activation_function"] = opt.FusedActivationFunction()
    attrs["merge_outputs"] = opt.MergeOutputs()
    attrs["proj_clip"] = opt.ProjClip()
    attrs["time_major"] = opt.TimeMajor()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_BidirectionalSequenceRNNOptions(_options: Any) -> dict[str, Any]:
    """Parse BidirectionalSequenceRNNOptions attributes."""
    from tflite.BidirectionalSequenceRNNOptions import BidirectionalSequenceRNNOptions

    opt = BidirectionalSequenceRNNOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["asymmetric_quantize_inputs"] = opt.AsymmetricQuantizeInputs()
    attrs["fused_activation_function"] = opt.FusedActivationFunction()
    attrs["merge_outputs"] = opt.MergeOutputs()
    attrs["time_major"] = opt.TimeMajor()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_BucketizeOptions(_options: Any) -> dict[str, Any]:
    """Parse BucketizeOptions attributes."""
    from tflite.BucketizeOptions import BucketizeOptions

    opt = BucketizeOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["boundaries"] = opt.BoundariesAsNumpy()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_ConcatEmbeddingsOptions(_options: Any) -> dict[str, Any]:
    """Parse ConcatEmbeddingsOptions attributes."""
    from tflite.ConcatEmbeddingsOptions import ConcatEmbeddingsOptions

    opt = ConcatEmbeddingsOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["num_channels"] = opt.NumChannels()
    attrs["embedding_dim_per_channel"] = opt.EmbeddingDimPerChannelAsNumpy()
    attrs["num_columns_per_channel"] = opt.NumColumnsPerChannelAsNumpy()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_CumsumOptions(_options: Any) -> dict[str, Any]:
    """Parse CumsumOptions attributes."""
    from tflite.CumsumOptions import CumsumOptions

    opt = CumsumOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["exclusive"] = opt.Exclusive()
    attrs["reverse"] = opt.Reverse()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_DensifyOptions(_options: Any) -> dict[str, Any]:
    """Parse DensifyOptions attributes."""
    return {}


def parse_DepthToSpaceOptions(_options: Any) -> dict[str, Any]:
    """Parse DepthToSpaceOptions attributes."""
    from tflite.DepthToSpaceOptions import DepthToSpaceOptions

    opt = DepthToSpaceOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["block_size"] = opt.BlockSize()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_DequantizeOptions(_options: Any) -> dict[str, Any]:
    """Parse DequantizeOptions attributes."""
    return {}


def parse_DynamicUpdateSliceOptions(_options: Any) -> dict[str, Any]:
    """Parse DynamicUpdateSliceOptions attributes."""
    return {}


def parse_EmbeddingLookupSparseOptions(_options: Any) -> dict[str, Any]:
    """Parse EmbeddingLookupSparseOptions attributes."""
    from tflite.EmbeddingLookupSparseOptions import EmbeddingLookupSparseOptions

    opt = EmbeddingLookupSparseOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["combiner"] = opt.Combiner()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_FakeQuantOptions(_options: Any) -> dict[str, Any]:
    """Parse FakeQuantOptions attributes."""
    from tflite.FakeQuantOptions import FakeQuantOptions

    opt = FakeQuantOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["max"] = opt.Max()
    attrs["min"] = opt.Min()
    attrs["narrow_range"] = opt.NarrowRange()
    attrs["num_bits"] = opt.NumBits()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_HashtableFindOptions(_options: Any) -> dict[str, Any]:
    """Parse HashtableFindOptions attributes."""
    return {}


def parse_HashtableImportOptions(_options: Any) -> dict[str, Any]:
    """Parse HashtableImportOptions attributes."""
    return {}


def parse_HashtableOptions(_options: Any) -> dict[str, Any]:
    """Parse HashtableOptions attributes."""
    from tflite.HashtableOptions import HashtableOptions

    opt = HashtableOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["key_dtype"] = opt.KeyDtype()
    attrs["table_id"] = opt.TableId()
    attrs["value_dtype"] = opt.ValueDtype()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_HashtableSizeOptions(_options: Any) -> dict[str, Any]:
    """Parse HashtableSizeOptions attributes."""
    return {}


def parse_LSHProjectionOptions(_options: Any) -> dict[str, Any]:
    """Parse LSHProjectionOptions attributes."""
    from tflite.LSHProjectionOptions import LSHProjectionOptions

    opt = LSHProjectionOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["type"] = opt.Type()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_LSTMOptions(_options: Any) -> dict[str, Any]:
    """Parse LSTMOptions attributes."""
    from tflite.LSTMOptions import LSTMOptions

    opt = LSTMOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["asymmetric_quantize_inputs"] = opt.AsymmetricQuantizeInputs()
    attrs["cell_clip"] = opt.CellClip()
    attrs["fused_activation_function"] = opt.FusedActivationFunction()
    attrs["kernel_type"] = opt.KernelType()
    attrs["proj_clip"] = opt.ProjClip()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_MatrixDiagOptions(_options: Any) -> dict[str, Any]:
    """Parse MatrixDiagOptions attributes."""
    return {}


def parse_MatrixSetDiagOptions(_options: Any) -> dict[str, Any]:
    """Parse MatrixSetDiagOptions attributes."""
    return {}


def parse_MirrorPadOptions(_options: Any) -> dict[str, Any]:
    """Parse MirrorPadOptions attributes."""
    from tflite.MirrorPadOptions import MirrorPadOptions

    opt = MirrorPadOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["mode"] = opt.Mode()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_NonMaxSuppressionV4Options(_options: Any) -> dict[str, Any]:
    """Parse NonMaxSuppressionV4Options attributes."""
    return {}


def parse_NonMaxSuppressionV5Options(_options: Any) -> dict[str, Any]:
    """Parse NonMaxSuppressionV5Options attributes."""
    return {}


def parse_PadOptions(_options: Any) -> dict[str, Any]:
    """Parse PadOptions attributes."""
    return {}


def parse_PadV2Options(_options: Any) -> dict[str, Any]:
    """Parse PadV2Options attributes."""
    return {}


def parse_QuantizeOptions(_options: Any) -> dict[str, Any]:
    """Parse QuantizeOptions attributes."""
    return {}


def parse_RNNOptions(_options: Any) -> dict[str, Any]:
    """Parse RNNOptions attributes."""
    from tflite.RNNOptions import RNNOptions

    opt = RNNOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["asymmetric_quantize_inputs"] = opt.AsymmetricQuantizeInputs()
    attrs["fused_activation_function"] = opt.FusedActivationFunction()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_RandomOptions(_options: Any) -> dict[str, Any]:
    """Parse RandomOptions attributes."""
    from tflite.RandomOptions import RandomOptions

    opt = RandomOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["seed"] = opt.Seed()
    attrs["seed2"] = opt.Seed2()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_RangeOptions(_options: Any) -> dict[str, Any]:
    """Parse RangeOptions attributes."""
    return {}


def parse_ReadVariableOptions(_options: Any) -> dict[str, Any]:
    """Parse ReadVariableOptions attributes."""
    return {}


def parse_ResizeBilinearOptions(_options: Any) -> dict[str, Any]:
    """Parse ResizeBilinearOptions attributes."""
    from tflite.ResizeBilinearOptions import ResizeBilinearOptions

    opt = ResizeBilinearOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["align_corners"] = opt.AlignCorners()
    attrs["half_pixel_centers"] = opt.HalfPixelCenters()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_ResizeNearestNeighborOptions(_options: Any) -> dict[str, Any]:
    """Parse ResizeNearestNeighborOptions attributes."""
    from tflite.ResizeNearestNeighborOptions import ResizeNearestNeighborOptions

    opt = ResizeNearestNeighborOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["align_corners"] = opt.AlignCorners()
    attrs["half_pixel_centers"] = opt.HalfPixelCenters()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_Rfft2dOptions(_options: Any) -> dict[str, Any]:
    """Parse Rfft2dOptions attributes."""
    return {}


def parse_SegmentSumOptions(_options: Any) -> dict[str, Any]:
    """Parse SegmentSumOptions attributes."""
    return {}


def parse_SequenceRNNOptions(_options: Any) -> dict[str, Any]:
    """Parse SequenceRNNOptions attributes."""
    from tflite.SequenceRNNOptions import SequenceRNNOptions

    opt = SequenceRNNOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["asymmetric_quantize_inputs"] = opt.AsymmetricQuantizeInputs()
    attrs["fused_activation_function"] = opt.FusedActivationFunction()
    attrs["time_major"] = opt.TimeMajor()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_SkipGramOptions(_options: Any) -> dict[str, Any]:
    """Parse SkipGramOptions attributes."""
    from tflite.SkipGramOptions import SkipGramOptions

    opt = SkipGramOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["include_all_ngrams"] = opt.IncludeAllNgrams()
    attrs["max_skip_size"] = opt.MaxSkipSize()
    attrs["ngram_size"] = opt.NgramSize()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_SpaceToBatchNDOptions(_options: Any) -> dict[str, Any]:
    """Parse SpaceToBatchNDOptions attributes."""
    return {}


def parse_SpaceToDepthOptions(_options: Any) -> dict[str, Any]:
    """Parse SpaceToDepthOptions attributes."""
    from tflite.SpaceToDepthOptions import SpaceToDepthOptions

    opt = SpaceToDepthOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["block_size"] = opt.BlockSize()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_SparseToDenseOptions(_options: Any) -> dict[str, Any]:
    """Parse SparseToDenseOptions attributes."""
    from tflite.SparseToDenseOptions import SparseToDenseOptions

    opt = SparseToDenseOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["validate_indices"] = opt.ValidateIndices()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_TopKV2Options(_options: Any) -> dict[str, Any]:
    """Parse TopKV2Options attributes."""
    return {}


def parse_UnidirectionalSequenceLSTMOptions(_options: Any) -> dict[str, Any]:
    """Parse UnidirectionalSequenceLSTMOptions attributes."""
    from tflite.UnidirectionalSequenceLSTMOptions import (
        UnidirectionalSequenceLSTMOptions,
    )

    opt = UnidirectionalSequenceLSTMOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["asymmetric_quantize_inputs"] = opt.AsymmetricQuantizeInputs()
    attrs["cell_clip"] = opt.CellClip()
    attrs["diagonal_recurrent_tensors"] = opt.DiagonalRecurrentTensors()
    attrs["fused_activation_function"] = opt.FusedActivationFunction()
    attrs["proj_clip"] = opt.ProjClip()
    attrs["time_major"] = opt.TimeMajor()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_UnsortedSegmentMaxOptions(_options: Any) -> dict[str, Any]:
    """Parse UnsortedSegmentMaxOptions attributes."""
    return {}


def parse_UnsortedSegmentMinOptions(_options: Any) -> dict[str, Any]:
    """Parse UnsortedSegmentMinOptions attributes."""
    return {}


def parse_UnsortedSegmentProdOptions(_options: Any) -> dict[str, Any]:
    """Parse UnsortedSegmentProdOptions attributes."""
    return {}


def parse_UnsortedSegmentSumOptions(_options: Any) -> dict[str, Any]:
    """Parse UnsortedSegmentSumOptions attributes."""
    return {}


def parse_VarHandleOptions(_options: Any) -> dict[str, Any]:
    """Parse VarHandleOptions attributes."""
    from tflite.VarHandleOptions import VarHandleOptions

    opt = VarHandleOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["container"] = opt.Container()
    attrs["shared_name"] = opt.SharedName()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs
