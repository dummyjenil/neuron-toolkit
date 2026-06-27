"""TFLite options parser for arithmetic operations."""
# ruff: noqa: PLC0415, N802

from __future__ import annotations

from typing import Any


def parse_AddOptions(_options: Any) -> dict[str, Any]:
    """Parse AddOptions attributes."""
    from tflite.AddOptions import AddOptions

    opt = AddOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs['fused_activation_function'] = opt.FusedActivationFunction()
    attrs['pot_scale_int16'] = opt.PotScaleInt16()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs

def parse_DivOptions(_options: Any) -> dict[str, Any]:
    """Parse DivOptions attributes."""
    from tflite.DivOptions import DivOptions

    opt = DivOptions()
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

def parse_MulOptions(_options: Any) -> dict[str, Any]:
    """Parse MulOptions attributes."""
    from tflite.MulOptions import MulOptions

    opt = MulOptions()
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

def parse_SubOptions(_options: Any) -> dict[str, Any]:
    """Parse SubOptions attributes."""
    from tflite.SubOptions import SubOptions

    opt = SubOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs['fused_activation_function'] = opt.FusedActivationFunction()
    attrs['pot_scale_int16'] = opt.PotScaleInt16()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs
