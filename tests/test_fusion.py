import pytest
from neuron_toolkit import ONNXGraph, Pattern, ONNXQuery

def test_onnx_graph_entry_point(complex_model):
    # Test ONNXGraph initialization and basic methods
    graph = ONNXGraph(complex_model)
    
    # Test query()
    q = graph.query()
    assert isinstance(q, ONNXQuery)
    assert q.count() == 6
    
    # Test match()
    pat = Pattern.op("Relu")
    match = graph.match(pat)
    assert match is not None
    assert match.start.op_type == "Relu"
    
    # Test findall()
    matches = graph.findall(Pattern.op("Add"))
    assert len(matches) == 1

def test_query_fluent_fusion(complex_model):
    graph = ONNXGraph(complex_model)
    
    # Test select()
    matches = graph.query().op("Relu").select(Pattern.op("Relu"))
    assert len(matches) == 1
    
    # Test where()
    q = graph.query().where(Pattern.op("Relu"))
    assert q.count() == 1
    assert q.first().op_type == "Relu"
    
    # Test match.query
    match = graph.match(Pattern.op("Add"))
    assert match.query.first().op_type == "Add"
    assert match.subgraph.count() == 1

def test_pattern_native_fusion(complex_model):
    graph = ONNXGraph(complex_model)
    pat = Pattern.op("Relu")
    
    # Test Pattern.find(graph)
    match = pat.find(graph)
    assert match.start.op_type == "Relu"
    
    # Test Pattern.findall(graph)
    matches = pat.findall(graph)
    assert len(matches) == 1
    
    # Test Pattern.filter(graph)
    q = pat.filter(graph)
    assert q.count() == 1
    assert q.first().op_type == "Relu"

def test_graph_replace_fusion(simple_model):
    graph = ONNXGraph(simple_model)
    # (Identity + B) * C
    # Replace Identity with Relu
    pat = Pattern.op("Identity")
    rewriter = graph.replace(pat, "Relu")
    
    new_model = rewriter.build()
    new_graph = ONNXGraph(new_model)
    assert new_graph.query().op("Relu").count() == 1
    assert new_graph.query().op("Identity").count() == 0

def test_query_to_pattern(complex_model):
    graph = ONNXGraph(complex_model)
    q = graph.query().op("Relu")
    pat = q.to_pattern()
    
    assert pat.op_type == "Relu"
    # Verify the pattern actually works
    assert graph.match(pat) is not None

def test_sequential_dsl(simple_model):
    # (Identity + B) * C
    graph = ONNXGraph(simple_model)
    # Match Identity -> Add
    pat = Pattern.op("Identity").then("Add")
    match = graph.match(pat)
    assert match is not None
    assert match.start.op_type == "Add"
    assert match.nodes[0].op_type == "Identity"
    assert match.nodes[1].op_type == "Add"

def test_query_follow(simple_model):
    graph = ONNXGraph(simple_model)
    # Start at Identity, follow to Mul
    # (Identity + B) * C
    pat = Pattern.op("Mul", Pattern.op("Add", Pattern.op("Identity")))
    q = graph.query().op("Identity").follow(pat)
    assert q.count() == 1
    assert q.first().op_type == "Mul"

def test_graph_passes(simple_model):
    graph = ONNXGraph(simple_model)
    assert hasattr(graph, "passes")
    assert callable(graph.passes.fuse_conv_bn)
    # The example pass doesn't actually do anything useful on simple_model yet,
    # but we verify it returns a rewriter
    from neuron_toolkit.rewriter import GraphRewriter
    assert isinstance(graph.passes.fuse_conv_bn(), GraphRewriter)

def test_pattern_traversal():
    pat = Pattern.op("Add", Pattern.op("Mul"), Pattern.const(1.0))
    nodes = pat.nodes()
    assert len(nodes) == 3
    assert any(n.op_type == "Add" for n in nodes)
    assert any(n.op_type == "Mul" for n in nodes)
    assert pat.parents[0].op_type == "Mul"
