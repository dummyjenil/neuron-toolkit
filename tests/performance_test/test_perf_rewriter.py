"""Performance test suite for Graph Rewriting and transformation passes."""

from __future__ import annotations

from neuron_toolkit import NeuronGraph
from neuron_toolkit.pattern import Pattern

from .perf_utils import (
    PerformanceTracker,
    create_large_onnx_model,
)


def test_perf_rewriter_bulk_replace_nodes():
    """Benchmark replacing 1,000 Relu operators with PRelu across a 4,000-node graph."""
    num_blocks = 1000
    model_proto = create_large_onnx_model(num_blocks=num_blocks)
    graph = NeuronGraph(model_proto)

    pat = Pattern.op("Relu", Pattern.any())

    with PerformanceTracker(
        op_name="NeuronGraph Bulk Pattern Replace (Relu -> PRelu)",
        scale_info=f"Replacing {num_blocks} ops in a {num_blocks * 4} node graph",
    ) as perf:
        rewriter = graph.replace(pat, "PRelu")
        rewritten_graph = rewriter.build()

    new_nodes = rewritten_graph.nodes if hasattr(rewritten_graph, "nodes") else rewritten_graph.graph.node
    op_types = [getattr(n, "op_type", None) for n in new_nodes]

    assert "PRelu" in op_types
    assert perf.duration_sec > 0.001


def test_perf_rewriter_delete_nodes_scale():
    """Benchmark deleting 500 ops from a large graph."""
    num_blocks = 500
    model_proto = create_large_onnx_model(num_blocks=num_blocks)
    graph = NeuronGraph(model_proto)

    q = graph.query()
    bn_nodes = q.find_by_op_type("BatchNormalization").nodes

    with PerformanceTracker(
        op_name="GraphRewriter Delete Nodes",
        scale_info=f"Deleting {len(bn_nodes)} BatchNormalization nodes",
    ) as perf:
        rw = graph.rewriter()
        rw.delete(bn_nodes)
        rw.build()

    assert perf.duration_sec > 0.0005
