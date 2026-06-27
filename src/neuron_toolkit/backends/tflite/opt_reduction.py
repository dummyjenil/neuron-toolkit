"""TFLite options parser for reduction operations."""
# ruff: noqa: PLC0415, N802

from __future__ import annotations

from typing import Any


def parse_ArgMaxOptions(_options: Any) -> dict[str, Any]:
    """Parse ArgMaxOptions attributes."""
    from tflite.ArgMaxOptions import ArgMaxOptions

    opt = ArgMaxOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["output_type"] = opt.OutputType()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_ArgMinOptions(_options: Any) -> dict[str, Any]:
    """Parse ArgMinOptions attributes."""
    from tflite.ArgMinOptions import ArgMinOptions

    opt = ArgMinOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["output_type"] = opt.OutputType()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_ReducerOptions(_options: Any) -> dict[str, Any]:
    """Parse ReducerOptions attributes."""
    from tflite.ReducerOptions import ReducerOptions

    opt = ReducerOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["keep_dims"] = opt.KeepDims()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs
