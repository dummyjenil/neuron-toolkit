"""Performance test suite for Pattern Matching and PatternDetector engine."""

from __future__ import annotations

from neuron_toolkit import NeuronGraph
from neuron_toolkit.pattern import Pattern

from .perf_utils import (
    PerformanceTracker,
    create_large_onnx_model,
)


def test_perf_pattern_match_conv_bn_add_relu():
    """Benchmark matching deep nested patterns (Relu(Add(BN(Conv), any))) across 1,000 blocks."""
    num_blocks = 1000
    model_proto = create_large_onnx_model(num_blocks=num_blocks)
    graph = NeuronGraph(model_proto)

    # Pattern: Relu(Add(BatchNormalization(Conv)))
    pat = Pattern.op(
        "Relu",
        Pattern.op(
            "Add",
            Pattern.op(
                "BatchNormalization",
                Pattern.op("Conv").capture("conv"),
            ).capture("bn"),
        ).capture("add"),
    ).capture("relu")

    with PerformanceTracker(
        op_name="Pattern Matching (Conv -> BN -> Add -> Relu)",
        scale_info=f"Finding matches in a {num_blocks * 4} node graph",
    ) as perf:
        matches = graph.findall(pat)

    assert len(matches) == num_blocks
    assert perf.duration_sec > 0.001


def test_perf_pattern_match_simple_activations():
    """Benchmark matching simple activation patterns across 2,000 blocks."""
    num_blocks = 2000
    model_proto = create_large_onnx_model(num_blocks=num_blocks)
    graph = NeuronGraph(model_proto)

    pat = Pattern.op("Relu", Pattern.any())

    with PerformanceTracker(
        op_name="Pattern Matching (Relu activations)",
        scale_info=f"Scanning {num_blocks * 4} nodes for Relu ops",
    ) as perf:
        matches = graph.findall(pat)

    assert len(matches) == num_blocks
    assert perf.duration_sec > 0.0005
