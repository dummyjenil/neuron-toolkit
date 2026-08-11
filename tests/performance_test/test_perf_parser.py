"""Performance test suite for ONNX & TFLite parser backends."""

from __future__ import annotations

import pytest
from neuron_toolkit.backends.onnx.parser import ONNXParser
from neuron_toolkit.backends.tflite.parser import TFLiteParser
from .perf_utils import (
    PerformanceTracker,
    create_large_onnx_model,
    create_large_tflite_model_bytes,
)


def test_perf_onnx_parser_large_scale():
    """Benchmark ONNXParser parsing and graph indexing overhead on a 1,500-block graph."""
    num_blocks = 1500
    model_proto = create_large_onnx_model(num_blocks=num_blocks)

    with PerformanceTracker(
        op_name="ONNXParser Raw Parse & Build Index",
        scale_info=f"{num_blocks} blocks (~{num_blocks * 4} nodes, ~{num_blocks * 5} initializers)",
    ) as perf:
        parser = ONNXParser(model_proto)

    assert len(parser.nodes) >= num_blocks * 4
    assert len(parser.tensor_map) >= num_blocks * 5
    assert perf.duration_sec > 0.001


def test_perf_tflite_parser_large_scale():
    """Benchmark TFLiteParser parsing overhead on a 1,500-operator flatbuffer."""
    num_layers = 1500
    tflite_bytes = create_large_tflite_model_bytes(num_layers=num_layers)

    with PerformanceTracker(
        op_name="TFLiteParser Flatbuffer Parsing",
        scale_info=f"{num_layers} operators and tensors",
    ) as perf:
        parser = TFLiteParser(tflite_bytes)

    assert len(parser.nodes) >= num_layers
    assert perf.duration_sec > 0.0005


def test_perf_parser_memory_retention():
    """Measure peak memory and allocation footprint when repeatedly parsing large models."""
    num_blocks = 500
    model_proto = create_large_onnx_model(num_blocks=num_blocks)

    with PerformanceTracker(
        op_name="ONNXParser Repeated Parse (5 iterations)",
        scale_info=f"5 iterations of {num_blocks} blocks",
    ) as perf:
        parsers = []
        for _ in range(5):
            parsers.append(ONNXParser(model_proto))

    assert len(parsers) == 5
    assert perf.peak_memory_mb > 0.5
