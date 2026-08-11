"""Performance test suite for Graph Loading, Slicing, and Traversal operations."""

from __future__ import annotations

import pytest
from neuron_toolkit import NeuronGraph
from .perf_utils import (
    PerformanceTracker,
    create_large_onnx_model,
    create_large_tflite_model_bytes,
)


def test_perf_graph_init_large_onnx():
    """Benchmark initialization time and memory for a 2,000-block (~8,000 node) ONNX graph."""
    num_blocks = 2000
    model_proto = create_large_onnx_model(num_blocks=num_blocks)

    with PerformanceTracker(
        op_name="NeuronGraph Init (ONNX)",
        scale_info=f"{num_blocks} blocks (~{num_blocks * 4} nodes)",
    ) as perf:
        graph = NeuronGraph(model_proto)

    assert len(graph.nodes) >= num_blocks * 4
    assert perf.duration_sec > 0.001
    assert perf.peak_memory_mb > 0.1


def test_perf_graph_init_large_tflite():
    """Benchmark initialization time and memory for a 1,000-node TFLite flatbuffer."""
    num_layers = 1000
    tflite_bytes = create_large_tflite_model_bytes(num_layers=num_layers)

    with PerformanceTracker(
        op_name="NeuronGraph Init (TFLite)",
        scale_info=f"{num_layers} operators",
    ) as perf:
        graph = NeuronGraph(tflite_bytes)

    assert len(graph.nodes) == num_layers
    assert perf.duration_sec > 0.0005


def test_perf_graph_slicing_deep():
    """Benchmark graph slicing across a deep ONNX network graph (1,000 blocks / 4,000 nodes)."""
    num_blocks = 1000
    model_proto = create_large_onnx_model(num_blocks=num_blocks)
    graph = NeuronGraph(model_proto)

    start_node = "relu_out_100"
    end_node = "relu_out_600"

    with PerformanceTracker(
        op_name="NeuronGraph Deep Slicing",
        scale_info=f"Slice between {start_node} and {end_node} (500 intermediate blocks)",
    ) as perf:
        sliced_graph = graph.slice(start_points=start_node, end_points=end_node)

    assert len(sliced_graph.nodes) > 0
    assert len(sliced_graph.nodes) < len(graph.nodes)
    assert perf.duration_sec > 0.0005
