"""TFLite options parser for nn operations."""
# ruff: noqa: PLC0415, N802

from __future__ import annotations

from typing import Any


def parse_BatchMatMulOptions(_options: Any) -> dict[str, Any]:
    """Parse BatchMatMulOptions attributes."""
    from tflite.BatchMatMulOptions import BatchMatMulOptions

    opt = BatchMatMulOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs['adj_x'] = opt.AdjX()
    attrs['adj_y'] = opt.AdjY()
    attrs['asymmetric_quantize_inputs'] = opt.AsymmetricQuantizeInputs()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs

def parse_Conv2DOptions(_options: Any) -> dict[str, Any]:
    """Parse Conv2DOptions attributes."""
    from tflite.Conv2DOptions import Conv2DOptions

    opt = Conv2DOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs['dilation_h_factor'] = opt.DilationHFactor()
    attrs['dilation_w_factor'] = opt.DilationWFactor()
    attrs['fused_activation_function'] = opt.FusedActivationFunction()
    attrs['padding'] = opt.Padding()
    attrs['quantized_bias_type'] = opt.QuantizedBiasType()
    attrs['stride_h'] = opt.StrideH()
    attrs['stride_w'] = opt.StrideW()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs

def parse_Conv3DOptions(_options: Any) -> dict[str, Any]:
    """Parse Conv3DOptions attributes."""
    from tflite.Conv3DOptions import Conv3DOptions

    opt = Conv3DOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs['dilation_d_factor'] = opt.DilationDFactor()
    attrs['dilation_h_factor'] = opt.DilationHFactor()
    attrs['dilation_w_factor'] = opt.DilationWFactor()
    attrs['fused_activation_function'] = opt.FusedActivationFunction()
    attrs['padding'] = opt.Padding()
    attrs['stride_d'] = opt.StrideD()
    attrs['stride_h'] = opt.StrideH()
    attrs['stride_w'] = opt.StrideW()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs

def parse_DepthwiseConv2DOptions(_options: Any) -> dict[str, Any]:
    """Parse DepthwiseConv2DOptions attributes."""
    from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions

    opt = DepthwiseConv2DOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs['depth_multiplier'] = opt.DepthMultiplier()
    attrs['dilation_h_factor'] = opt.DilationHFactor()
    attrs['dilation_w_factor'] = opt.DilationWFactor()
    attrs['fused_activation_function'] = opt.FusedActivationFunction()
    attrs['padding'] = opt.Padding()
    attrs['stride_h'] = opt.StrideH()
    attrs['stride_w'] = opt.StrideW()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs

def parse_FullyConnectedOptions(_options: Any) -> dict[str, Any]:
    """Parse FullyConnectedOptions attributes."""
    from tflite.FullyConnectedOptions import FullyConnectedOptions

    opt = FullyConnectedOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs['asymmetric_quantize_inputs'] = opt.AsymmetricQuantizeInputs()
    attrs['fused_activation_function'] = opt.FusedActivationFunction()
    attrs['keep_num_dims'] = opt.KeepNumDims()
    attrs['quantized_bias_type'] = opt.QuantizedBiasType()
    attrs['weights_format'] = opt.WeightsFormat()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs

def parse_L2NormOptions(_options: Any) -> dict[str, Any]:
    """Parse L2NormOptions attributes."""
    from tflite.L2NormOptions import L2NormOptions

    opt = L2NormOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs['fused_activation_function'] = opt.FusedActivationFunction()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs

def parse_LocalResponseNormalizationOptions(_options: Any) -> dict[str, Any]:
    """Parse LocalResponseNormalizationOptions attributes."""
    from tflite.LocalResponseNormalizationOptions import (
        LocalResponseNormalizationOptions,
    )

    opt = LocalResponseNormalizationOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs['alpha'] = opt.Alpha()
    attrs['beta'] = opt.Beta()
    attrs['bias'] = opt.Bias()
    attrs['radius'] = opt.Radius()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs

def parse_Pool2DOptions(_options: Any) -> dict[str, Any]:
    """Parse Pool2DOptions attributes."""
    from tflite.Pool2DOptions import Pool2DOptions

    opt = Pool2DOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs['filter_height'] = opt.FilterHeight()
    attrs['filter_width'] = opt.FilterWidth()
    attrs['fused_activation_function'] = opt.FusedActivationFunction()
    attrs['padding'] = opt.Padding()
    attrs['stride_h'] = opt.StrideH()
    attrs['stride_w'] = opt.StrideW()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs

def parse_SVDFOptions(_options: Any) -> dict[str, Any]:
    """Parse SVDFOptions attributes."""
    from tflite.SVDFOptions import SVDFOptions

    opt = SVDFOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs['asymmetric_quantize_inputs'] = opt.AsymmetricQuantizeInputs()
    attrs['fused_activation_function'] = opt.FusedActivationFunction()
    attrs['rank'] = opt.Rank()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs

def parse_TransposeConvOptions(_options: Any) -> dict[str, Any]:
    """Parse TransposeConvOptions attributes."""
    from tflite.TransposeConvOptions import TransposeConvOptions

    opt = TransposeConvOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs['fused_activation_function'] = opt.FusedActivationFunction()
    attrs['padding'] = opt.Padding()
    attrs['quantized_bias_type'] = opt.QuantizedBiasType()
    attrs['stride_h'] = opt.StrideH()
    attrs['stride_w'] = opt.StrideW()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs
