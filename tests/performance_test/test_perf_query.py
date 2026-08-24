"""Performance test suite for Graph Query operations (filters, traversals, and set logic)."""

from __future__ import annotations

from neuron_toolkit import NeuronGraph

from .perf_utils import (
    PerformanceTracker,
    create_large_onnx_model,
)


def test_perf_query_by_op_type_scale():
    """Benchmark querying by op_type on an 8,000-node graph."""
    num_blocks = 2000
    model_proto = create_large_onnx_model(num_blocks=num_blocks)
    graph = NeuronGraph(model_proto)

    with PerformanceTracker(
        op_name="GraphQuery find_by_op_type",
        scale_info=f"{num_blocks * 4} nodes in graph",
    ) as perf:
        conv_query = graph.query().find_by_op_type("Conv")
        conv_count = conv_query.count()

    assert conv_count == num_blocks
    assert perf.duration_sec > 0.0001


def test_perf_query_set_operations_scale():
    """Benchmark set operations (Union, Intersection) over large query sets (2,000+ results)."""
    num_blocks = 2000
    model_proto = create_large_onnx_model(num_blocks=num_blocks)
    graph = NeuronGraph(model_proto)
    q = graph.query()

    with PerformanceTracker(
        op_name="GraphQuery Set Operations (Union & Intersect)",
        scale_info=f"Union and Intersection across {num_blocks * 4} graph nodes",
    ) as perf:
        q_conv = q.find_by_op_type("Conv")
        q_bn = q.find_by_op_type("BatchNormalization")
        q_add = q.find_by_op_type("Add")

        q_union = q_conv | q_bn | q_add
        union_count = q_union.count()

        q_intersect = q_union & q_conv
        intersect_count = q_intersect.count()

    assert union_count == num_blocks * 3
    assert intersect_count == num_blocks
    assert perf.duration_sec > 0.0001


def test_perf_query_descendants_traversal_deep():
    """Benchmark graph descendant traversal from the root across a 2,000-deep node chain."""
    num_blocks = 2000
    model_proto = create_large_onnx_model(num_blocks=num_blocks)
    graph = NeuronGraph(model_proto)
    q = graph.query()

    root_conv = q.find_by_name("conv_0", exact=True)

    with PerformanceTracker(
        op_name="GraphQuery Ancestor/Descendant Deep Traversal",
        scale_info=f"Traversing descendants across {num_blocks * 4} nodes",
    ) as perf:
        descendants = root_conv.descendants()
        desc_count = descendants.count()

    assert desc_count >= num_blocks * 3
    assert perf.duration_sec > 0.001
