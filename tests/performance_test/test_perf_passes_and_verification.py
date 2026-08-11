"""Performance test suite for Optimization Passes (GraphPasses) and PyTorch Verification/Slicing."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from neuron_toolkit import NeuronGraph
from neuron_toolkit._utils import trace_subgraph_boundaries
from .perf_utils import (
    PerformanceTracker,
    create_large_onnx_model,
)

pytest.importorskip("onnxruntime")


def test_perf_fuse_conv_bn_pass():
    """Benchmark built-in Conv+BatchNormalization fusion pass across 1,000 blocks."""
    num_blocks = 1000
    model_proto = create_large_onnx_model(num_blocks=num_blocks)
    graph = NeuronGraph(model_proto)

    with PerformanceTracker(
        op_name="GraphPasses fuse_conv_bn() Optimization Pass",
        scale_info=f"Folding BatchNormalization into Conv across {num_blocks} blocks",
    ) as perf:
        fused_rewriter = graph.passes.fuse_conv_bn()
        fused_model = fused_rewriter.build()

    new_graph = NeuronGraph(fused_model)
    bn_nodes = new_graph.query().find_by_op_type("BatchNormalization").count()

    assert bn_nodes == 0
    assert perf.duration_sec > 0.001


def test_perf_trace_subgraph_boundaries_scale():
    """Benchmark trace_subgraph_boundaries NetworkX reachability algorithms on a 2,000-block graph."""
    num_blocks = 2000
    model_proto = create_large_onnx_model(num_blocks=num_blocks)
    graph = NeuronGraph(model_proto)

    nodes = graph.nodes
    tensor_map = graph.tensor_map

    with PerformanceTracker(
        op_name="trace_subgraph_boundaries Reachability Graph Traversal",
        scale_info=f"Tracing boundaries across {len(nodes)} nodes",
    ) as perf:
        kept_ops, new_inputs, new_outputs = trace_subgraph_boundaries(
            nodes=nodes,
            tensor_map=tensor_map,
            start_points="relu_out_100",
            end_points="relu_out_800",
        )

    assert len(kept_ops) > 0
    assert len(new_inputs) > 0
    assert len(new_outputs) > 0
    assert perf.duration_sec > 0.001


def test_perf_compare_outputs_pytorch_verification():
    """Benchmark compare_outputs numerical verification (PyTorch vs ONNXRuntime)."""
    # Simple PyTorch model for verification benchmark
    class ReluModule(nn.Module):
        def forward(self, x):
            return torch.relu(x)

    py_model = ReluModule().eval()
    input_np = np.random.randn(1, 3, 32, 32).astype(np.float32)

    # Build matching ONNX model directly
    from onnx import TensorProto, helper

    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 32, 32])
    relu_node = helper.make_node("Relu", ["input"], ["output"], name="relu0")
    graph_proto = helper.make_graph([relu_node], "relu_verify", [x_info], [y_info])
    onnx_model = helper.make_model(graph_proto)

    graph = NeuronGraph(onnx_model)

    with PerformanceTracker(
        op_name="compare_outputs PyTorch Numerical Cross-Verification",
        scale_info="Executing PyTorch vs ONNXRuntime inference & computing MAE/MSE/Cosine Similarity",
    ) as perf:
        report = graph.compare_outputs(
            start_points="input",
            end_points="output",
            pytorch_module=py_model,
            inputs_dict={"input": input_np},
        )

    assert len(report) > 0
    first_metric = next(iter(report.values()))
    assert first_metric["all_close"] is True
    assert first_metric["cosine_similarity"] > 0.99
    assert perf.duration_sec > 0.0005
