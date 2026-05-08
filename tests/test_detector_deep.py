
import pytest
import numpy as np
from onnx import helper, TensorProto
from onnx_toolkit.parser import ONNXParser
from onnx_toolkit.pattern import Pattern, PatternDetector

@pytest.fixture
def branching_model(tmp_path):
    # A -> Relu -> Add(B) -> Mul(C) -> Out
    #             |
    #             -> Add(D) -> Out2
    
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 5])
    B_val = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    B = helper.make_tensor("B", TensorProto.FLOAT, [5], B_val.tobytes(), raw=True)
    C_val = np.array([2, 2, 2, 2, 2], dtype=np.float32)
    C = helper.make_tensor("C", TensorProto.FLOAT, [5], C_val.tobytes(), raw=True)
    D_val = np.array([3, 3, 3, 3, 3], dtype=np.float32)
    D = helper.make_tensor("D", TensorProto.FLOAT, [5], D_val.tobytes(), raw=True)
    
    node0 = helper.make_node("Relu", ["A"], ["relu_out"], name="n0")
    node1 = helper.make_node("Add", ["relu_out", "B"], ["add1_out"], name="n1")
    node2 = helper.make_node("Mul", ["add1_out", "C"], ["output"], name="n2")
    node3 = helper.make_node("Add", ["relu_out", "D"], ["output2"], name="n3")
    
    out1 = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5])
    out2 = helper.make_tensor_value_info("output2", TensorProto.FLOAT, [1, 5])
    
    graph = helper.make_graph(
        [node0, node1, node2, node3],
        "branching",
        [A],
        [out1, out2],
        [B, C, D]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = str(tmp_path / "branching.onnx")
    import onnx
    onnx.save(model, path)
    return path

def test_detector_basic_match(branching_model):
    parser = ONNXParser(branching_model)
    # Match Mul(Add(Relu, const), const)
    p = Pattern.op("Mul", 
        Pattern.op("Add", 
            Pattern.op("Relu").capture("r"),
            Pattern.const(1.0)
        ).capture("a"),
        Pattern.const(2.0)
    ).capture("m")
    
    match = parser.pattern_detect(p, start_node="n2")
    assert match is not None
    assert match.bindings["m"].name == "n2"
    assert match.bindings["a"].name == "n1"
    assert match.bindings["r"].name == "n0"
    assert len(match.nodes) == 3

def test_detector_find_all(branching_model):
    parser = ONNXParser(branching_model)
    # Find all Add nodes
    p = Pattern.op("Add")
    det = PatternDetector(parser.model)
    matches = det.find_all(p)
    assert len(matches) == 2
    names = {m.start.name for m in matches}
    assert names == {"n1", "n3"}

def test_detector_commutative(branching_model):
    parser = ONNXParser(branching_model)
    # Add(const, Relu) should match n1 even though B is second input in proto
    p = Pattern.op("Add", Pattern.const(1.0), Pattern.op("Relu"))
    match = parser.pattern_detect(p, start_node="n1")
    assert match is not None

def test_detector_attribute_constraints(tmp_path):
    # Model with Conv(kernel_shape=[3,3])
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
    W = helper.make_tensor("W", TensorProto.FLOAT, [1, 1, 3, 3], np.zeros((1, 1, 3, 3), dtype=np.float32).tobytes(), raw=True)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 3, 3])
    node = helper.make_node("Conv", ["X", "W"], ["Y"], name="conv", kernel_shape=[3, 3])
    graph = helper.make_graph([node], "attr_test", [X], [Y], [W])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = str(tmp_path / "attr.onnx")
    import onnx
    onnx.save(model, path)
    
    parser = ONNXParser(path)
    
    # Correct constraint
    p = Pattern.op("Conv").where(kernel_shape=[3, 3])
    assert parser.pattern_detect(p, start_node="conv") is not None
    
    # Incorrect constraint
    p = Pattern.op("Conv").where(kernel_shape=[5, 5])
    assert parser.pattern_detect(p, start_node="conv") is None
    
    # Callable constraint
    p = Pattern.op("Conv").where(kernel_shape=lambda x: x[0] == 3)
    assert parser.pattern_detect(p, start_node="conv") is not None

def test_detector_shape_dtype_constraints(branching_model):
    parser = ONNXParser(branching_model)
    # n2 output shape is [1, 5], dtype is float32
    p = Pattern.op("Mul").with_output_rank(2).with_dtype("float32")
    assert parser.pattern_detect(p, start_node="n2") is not None
    
    p = Pattern.op("Mul").with_output_rank(3)
    assert parser.pattern_detect(p, start_node="n2") is None

def test_detector_any_of(branching_model):
    parser = ONNXParser(branching_model)
    # Match either Add or Mul
    p = Pattern.any_of(Pattern.op("Add"), Pattern.op("Mul"))
    assert parser.pattern_detect(p, start_node="n1") is not None
    assert parser.pattern_detect(p, start_node="n2") is not None
    assert parser.pattern_detect(p, start_node="n0") is None

def test_detector_wildcard(branching_model):
    parser = ONNXParser(branching_model)
    # any() matches any node
    p = Pattern.any()
    assert parser.pattern_detect(p, start_node="n0") is not None
    assert parser.pattern_detect(p, start_node="n1") is not None

def test_detector_constant_node(tmp_path):
    # Model with a Constant node instead of an initializer
    val = np.array([42.0], dtype=np.float32)
    node_const = helper.make_node(
        "Constant",
        [],
        ["const_out"],
        value=helper.make_tensor("value", TensorProto.FLOAT, [1], val.flatten())
    )
    node_add = helper.make_node("Add", ["X", "const_out"], ["Y"], name="add")
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    graph = helper.make_graph([node_const, node_add], "const_node_test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = str(tmp_path / "const_node.onnx")
    import onnx
    onnx.save(model, path)
    
    parser = ONNXParser(path)
    # _match_const should handle Constant nodes
    p = Pattern.op("Add", Pattern.any(), Pattern.const(42.0))
    # Wait, Add inputs are [X, const_out]. 
    # _parents will see const_out as being produced by node_const.
    # _dfs for Pattern.const(42.0) will call _match_const(node_const, 42.0).
    assert parser.pattern_detect(p, start_node="add") is not None

def test_detector_no_match_if_too_few_parents(branching_model):
    parser = ONNXParser(branching_model)
    # n0 (Relu) has only 1 input (graph input), so it has 0 parents in _parents
    # Pattern expects 1 parent node
    p = Pattern.op("Relu", Pattern.any())
    assert parser.pattern_detect(p, start_node="n0") is None
