import numpy as np

from onnx_toolkit.parser import ONNXParser
from onnx_toolkit.pattern import Pattern


def test_pattern_dsl():
    p = Pattern.op("Add", Pattern.any(), Pattern.const(1.0))
    assert p.op_type == "Add"
    assert len(p.inputs) == 2
    assert p.inputs[0].op_type == "__any__"
    assert p.inputs[1].op_type == "__const__"
    assert p.inputs[1].value == 1.0

def test_pattern_arithmetic_dsl():
    x = Pattern.any()
    p = (x + 1.0) * 2.0
    assert p.op_type == "Mul"
    assert p.inputs[0].op_type == "Add"
    assert p.inputs[1].value == 2.0
    assert p.inputs[0].inputs[0] is x
    assert p.inputs[0].inputs[1].value == 1.0

def test_pattern_matching_simple(simple_model):
    parser = ONNXParser(simple_model)

    # Match (Identity + B)
    B_val = np.array([[1, 2, 3]], dtype=np.float32)
    pat = Pattern.op("Add", Pattern.any(), Pattern.const(B_val))
    res = parser.pattern_detect(pat, start_node="node_add")
    assert res is not None
    assert res.start.name == "node_add"

    # Match (Identity + B) * C
    C_val = np.array([[2, 2, 2]], dtype=np.float32)
    pat_full = Pattern.op("Mul", pat, Pattern.const(C_val))
    res_full = parser.pattern_detect(pat_full, start_node="node_mul")
    assert res_full is not None
    assert len(res_full.nodes) == 3

def test_pattern_capture(simple_model):
    parser = ONNXParser(simple_model)

    B_val = np.array([[1, 2, 3]], dtype=np.float32)
    pat = Pattern.op("Add", Pattern.any().capture("input_a"), Pattern.const(B_val).capture("input_b")).capture("add_node")
    res = parser.pattern_detect(pat, start_node="node_add")
    assert res is not None
    assert "add_node" in res.bindings
    assert res.bindings["add_node"].name == "node_add"
    assert res.bindings["input_a"].name == "node_id"

def test_pattern_any_of(complex_model):
    parser = ONNXParser(complex_model)

    # Match either Sigmoid or Tanh
    pat = Pattern.any_of(Pattern.op("Sigmoid", Pattern.any()), Pattern.op("Tanh", Pattern.any()))

    res1 = parser.pattern_detect(pat, start_node="n4")
    assert res1 is not None
    assert res1.start.op_type == "Sigmoid"

    res2 = parser.pattern_detect(pat, start_node="n5")
    assert res2 is not None
    assert res2.start.op_type == "Tanh"

def test_pattern_commutative(simple_model):
    parser = ONNXParser(simple_model)

    # Add(A, B) should match even if we swap patterns in DSL if we use commutative logic
    # In detector.py, Add and Mul are marked as commutative

    # B is const [[1,2,3]]
    B_val = np.array([[1, 2, 3]], dtype=np.float32)

    # Pattern: Add(const, any)
    pat = Pattern.op("Add", Pattern.const(B_val), Pattern.any())
    res = parser.pattern_detect(pat, start_node="node_add")
    assert res is not None # Should match because Add is commutative
