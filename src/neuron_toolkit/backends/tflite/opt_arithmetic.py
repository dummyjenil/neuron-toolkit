"""TFLite options parser for arithmetic operations."""
# ruff: noqa: PLC0415, N802

from __future__ import annotations

from typing import Any


def parse_ATan2Options(_options: Any) -> dict[str, Any]:
    """Parse ATan2Options attributes."""
    return {}


def parse_AbsOptions(_options: Any) -> dict[str, Any]:
    """Parse AbsOptions attributes."""
    return {}


def parse_AddNOptions(_options: Any) -> dict[str, Any]:
    """Parse AddNOptions attributes."""
    return {}


def parse_AddOptions(_options: Any) -> dict[str, Any]:
    """Parse AddOptions attributes."""
    from tflite.AddOptions import AddOptions

    opt = AddOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["fused_activation_function"] = opt.FusedActivationFunction()
    attrs["pot_scale_int16"] = opt.PotScaleInt16()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_BitwiseXorOptions(_options: Any) -> dict[str, Any]:
    """Parse BitwiseXorOptions attributes."""
    return {}


def parse_CosOptions(_options: Any) -> dict[str, Any]:
    """Parse CosOptions attributes."""
    return {}


def parse_DivOptions(_options: Any) -> dict[str, Any]:
    """Parse DivOptions attributes."""
    from tflite.DivOptions import DivOptions

    opt = DivOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["fused_activation_function"] = opt.FusedActivationFunction()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_ExpOptions(_options: Any) -> dict[str, Any]:
    """Parse ExpOptions attributes."""
    return {}


def parse_FloorDivOptions(_options: Any) -> dict[str, Any]:
    """Parse FloorDivOptions attributes."""
    return {}


def parse_FloorModOptions(_options: Any) -> dict[str, Any]:
    """Parse FloorModOptions attributes."""
    return {}


def parse_LogSoftmaxOptions(_options: Any) -> dict[str, Any]:
    """Parse LogSoftmaxOptions attributes."""
    return {}


def parse_MaximumMinimumOptions(_options: Any) -> dict[str, Any]:
    """Parse MaximumMinimumOptions attributes."""
    return {}


def parse_MulOptions(_options: Any) -> dict[str, Any]:
    """Parse MulOptions attributes."""
    from tflite.MulOptions import MulOptions

    opt = MulOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["fused_activation_function"] = opt.FusedActivationFunction()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_NegOptions(_options: Any) -> dict[str, Any]:
    """Parse NegOptions attributes."""
    return {}


def parse_PowOptions(_options: Any) -> dict[str, Any]:
    """Parse PowOptions attributes."""
    return {}


def parse_RightShiftOptions(_options: Any) -> dict[str, Any]:
    """Parse RightShiftOptions attributes."""
    return {}


def parse_SignOptions(_options: Any) -> dict[str, Any]:
    """Parse SignOptions attributes."""
    return {}


def parse_SquareOptions(_options: Any) -> dict[str, Any]:
    """Parse SquareOptions attributes."""
    return {}


def parse_SquaredDifferenceOptions(_options: Any) -> dict[str, Any]:
    """Parse SquaredDifferenceOptions attributes."""
    return {}


def parse_SubOptions(_options: Any) -> dict[str, Any]:
    """Parse SubOptions attributes."""
    from tflite.SubOptions import SubOptions

    opt = SubOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["fused_activation_function"] = opt.FusedActivationFunction()
    attrs["pot_scale_int16"] = opt.PotScaleInt16()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs
