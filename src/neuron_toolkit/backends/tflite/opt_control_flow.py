"""TFLite options parser for control_flow operations."""
# ruff: noqa: PLC0415, N802

from __future__ import annotations

from typing import Any


def parse_CallOnceOptions(_options: Any) -> dict[str, Any]:
    """Parse CallOnceOptions attributes."""
    from tflite.CallOnceOptions import CallOnceOptions

    opt = CallOnceOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["init_subgraph_index"] = opt.InitSubgraphIndex()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_CallOptions(_options: Any) -> dict[str, Any]:
    """Parse CallOptions attributes."""
    from tflite.CallOptions import CallOptions

    opt = CallOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["subgraph"] = opt.Subgraph()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_IfOptions(_options: Any) -> dict[str, Any]:
    """Parse IfOptions attributes."""
    from tflite.IfOptions import IfOptions

    opt = IfOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["else_subgraph_index"] = opt.ElseSubgraphIndex()
    attrs["then_subgraph_index"] = opt.ThenSubgraphIndex()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_WhileOptions(_options: Any) -> dict[str, Any]:
    """Parse WhileOptions attributes."""
    from tflite.WhileOptions import WhileOptions

    opt = WhileOptions()
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
