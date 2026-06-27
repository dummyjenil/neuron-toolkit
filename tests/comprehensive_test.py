"""Comprehensive, practical tests for neuron-toolkit.

This test suite covers the end-to-end features of neuron-toolkit, serving as both
a correctness verification and a practical guide/tutorial for developers.
"""

import os
import numpy as np
import pytest

from neuron_toolkit import NeuronGraph, Pattern, NeuronRewriter


def test_neuron_graph_basics(simple_model):
    """1. Graph Loading and Basic Inspection API.

    Demonstrates how to load a model (ONNX/TFLite) using NeuronGraph and inspect its basic metadata.
    """
    # Load the graph using the file path
    graph = NeuronGraph(simple_model)

    # Inspect the basic attributes: nodes, initializers (tensor_map)
    assert len(graph.nodes) == 3
    assert "B" in graph.tensor_map
    assert "C" in graph.tensor_map

    # shape_info is a dictionary containing shape (rank) and data type metadata for each tensor
    assert "output" in graph.shape_info
    rank, dtype = graph.shape_info["output"]
    assert rank == 2
    assert dtype == "float32"


def test_neuron_query_engine(complex_model):
    """2. NeuronQuery Fluent API.

    Demonstrates the chainable filter methods, custom predicates, set operations,
    topological sorting, and graph traversals.
    """
    graph = NeuronGraph(complex_model)
    q = graph.query()

    # Filter by operation type (op_type)
    relu_nodes = q.op("Relu")
    assert relu_nodes.count() == 1
    assert relu_nodes.first().op_type == "Relu"

    # Filter by name (exact or substring)
    n2_node = q.name("n2", exact=True)
    assert n2_node.count() == 1
    assert n2_node.first().name == "n2"

    n_nodes = q.name("n")  # Substring matches "n0", "n1", "n2", "n3", "n4", "n5"
    assert n_nodes.count() == 6

    # Filter by rank and data type
    float_nodes = q.dtype("float32")
    assert float_nodes.count() > 0

    # Filter by nodes producing or consuming a specific tensor
    t_nodes = q.find_by_tensor("relu_out")
    # Matches Relu (producer) and Add (consumer)
    assert t_nodes.count() == 2

    # Chaining multiple filters
    chained = q.op("Add").name("n2")
    assert chained.count() == 1

    # Custom predicate filtering using .filter()
    multi_input_nodes = q.filter(lambda node: len(getattr(node, "input", [])) > 1)
    assert (
        multi_input_nodes.count() == 2
    )  # Add (takes relu_out & B) and Mul (takes add_out & C)

    # Set operations on queries (Union |, Intersection &, Difference -)
    relu_q = q.op("Relu")
    add_q = q.op("Add")

    union = relu_q | add_q
    assert union.count() == 2

    intersection = union & relu_q
    assert intersection.count() == 1

    difference = union - relu_q
    assert difference.count() == 1
    assert difference.first().op_type == "Add"

    # Topological sort checks
    sorted_q = q.topological_sort()
    assert sorted_q.count() == 6
    assert sorted_q.is_topologically_sorted()

    # Graph traversal (Parents, Children, Ancestors, Descendants)
    relu_node_q = q.name("n1", exact=True)
    add_node_q = q.name("n2", exact=True)

    # Immediate children and parents
    assert relu_node_q.children().first().name == "n2"
    assert add_node_q.parents().first().name == "n1"

    # Ancestors: all upstream nodes contributing to the target node
    anc = q.name("n4", exact=True).ancestors()
    assert set(n.name for n in anc) == {"n0", "n1", "n2", "n3"}

    # Descendants: all downstream nodes affected by the target node
    desc = relu_node_q.descendants()
    assert set(n.name for n in desc) == {"n2", "n3", "n4", "n5"}


def test_pattern_dsl_and_matching(complex_model, simple_model):
    """3. Pattern DSL and Detection.

    Shows how to build patterns, match them, and inspect the resulting subgraphs.
    """
    graph = NeuronGraph(complex_model)

    # Constructing a pattern: Mul taking Add as input
    pattern = Pattern.op("Mul", Pattern.op("Add"))

    # Match the pattern in the graph
    match = graph.match(pattern)
    assert match is not None
    assert match.start.op_type == "Mul"
    assert len(match.nodes) == 2
    assert match.nodes[0].op_type == "Add"
    assert match.nodes[1].op_type == "Mul"

    # Sequential pattern chaining using .then()
    # Match: Identity -> Relu
    seq_pattern = Pattern.op("Identity").then("Relu")
    seq_match = graph.match(seq_pattern)
    assert seq_match is not None
    assert seq_match.nodes[0].op_type == "Identity"
    assert seq_match.nodes[1].op_type == "Relu"

    # Arithmetic operator overloading DSL
    # Match (x + B) where B is a constant tensor in simple_model
    simple_graph = NeuronGraph(simple_model)
    B_val = simple_graph.tensor_map["B"]

    x = Pattern.any()
    pat = x + B_val
    res = simple_graph.match(pat)
    assert res is not None
    assert res.start.op_type == "Add"


def test_graph_rewriting_and_replacement(simple_model, tmp_path):
    """4. Graph Rewriting and Functional Modification.

    Shows how a developer can replace a matched pattern/subgraph with a fused node.
    """
    graph = NeuronGraph(simple_model)
    # The simple model structure: (Identity(A) + B) * C

    # Get the initializer B
    B_val = graph.tensor_map["B"]

    # Let's match the Identity followed by Add
    pattern = Pattern.op("Add", Pattern.op("Identity"), Pattern.const(B_val))

    # Replace all occurrences of this pattern with a custom fused node
    rewriter = graph.replace(pattern, "FusedAdd", name="my_fused_add")
    assert isinstance(rewriter, NeuronRewriter)

    # Build and save the new model
    output_model_path = os.path.join(tmp_path, "fused_model.onnx")
    rewriter.build(output_model_path)

    assert os.path.exists(output_model_path)

    # Verify the rewritten model structure
    new_graph = NeuronGraph(output_model_path)
    assert new_graph.query().op("Identity").count() == 0
    assert new_graph.query().op("Add").count() == 0

    fused_nodes = new_graph.query().op("FusedAdd")
    assert fused_nodes.count() == 1
    assert fused_nodes.first().name == "my_fused_add"


def test_manual_rewriter_api(simple_model, tmp_path):
    """5. Manual Rewriting API.

    Shows how to explicitly delete and insert nodes in the graph.
    """
    graph = NeuronGraph(simple_model)
    rewriter = graph.rewriter()

    # Find the Add node
    add_node = graph.query().op("Add").first()

    # Let's insert an Identity node before the Add node
    rewriter.insert_before(
        add_node,
        "Identity",
        inputs=["A"],
        outputs=["A_ident_new"],
        name="manual_identity",
    )

    # Build the modified model
    output_model_path = os.path.join(tmp_path, "manual_rewrite.onnx")
    rewriter.build(output_model_path)

    new_graph = NeuronGraph(output_model_path)
    # Total nodes should be 4 now
    assert len(new_graph.nodes) == 4

    ident_nodes = new_graph.query().name("manual_identity", exact=True)
    assert ident_nodes.count() == 1
