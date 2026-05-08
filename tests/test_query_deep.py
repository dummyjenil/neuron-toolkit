import numpy as np
import pytest
from onnx import TensorProto, helper

from onnx_toolkit.parser import ONNXParser


def test_query_basics(branching_model):
    parser = ONNXParser(branching_model)
    q = parser.find()

    assert len(q) == 4
    assert q.count() == 4

    # Filter by op type
    adds = q.op("Add")
    assert len(adds) == 2
    assert all(n.op_type == "Add" for n in adds)

    # Filter by name
    n0 = q.name("n0")
    assert len(n0) == 1
    assert n0.first().op_type == "Relu"


def test_query_navigation(branching_model):
    parser = ONNXParser(branching_model)
    q = parser.find()

    # n2 (Mul) inputs
    n2 = q.name("n2")
    inputs = n2.inputs()
    # n2 inputs are [add1_out, C]. C is an initializer.
    # ONNXQuery.inputs() usually returns producers of the inputs.
    assert len(inputs) == 1
    assert inputs.first().name == "n1"

    # n0 (Relu) outputs
    n0 = q.name("n0")
    outputs = n0.outputs()
    # n0 (relu_out) is input to n1 and n3
    assert len(outputs) == 2
    names = {n.name for n in outputs}
    assert names == {"n1", "n3"}


def test_query_attrs(tmp_path):
    # Model with Conv(kernel_shape=[3,3])
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
    W = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [1, 1, 3, 3],
        np.zeros((1, 1, 3, 3), dtype=np.float32).tobytes(),
        raw=True,
    )
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 3, 3])
    node = helper.make_node("Conv", ["X", "W"], ["Y"], name="conv", kernel_shape=[3, 3])
    graph = helper.make_graph([node], "attr_test", [X], [Y], [W])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = str(tmp_path / "attr_query.onnx")
    import onnx

    onnx.save(model, path)

    parser = ONNXParser(path)
    q = parser.find()

    assert q.attr("kernel_shape", [3, 3]).count() == 1
    assert q.attr("kernel_shape", [5, 5]).count() == 0
    assert q.attr("non_existent").count() == 0


def test_query_params(branching_model):
    parser = ONNXParser(branching_model)
    q = parser.find()

    # n1 (Add) has parameter B
    n1 = q.name("n1")
    assert n1.has_params().count() == 1
    params = n1.tensor()
    assert "B" in params
    assert params["B"].shape == (5,)


def test_query_single_node_tensor(branching_model):
    parser = ONNXParser(branching_model)
    n1 = parser.find().name("n1", exact=True)
    assert n1.single_node.op_type == "Add"
    assert n1.single_tensor.shape == (5,)


def test_query_set_ops(branching_model):
    parser = ONNXParser(branching_model)
    adds = parser.find().op("Add")
    n1 = parser.find().name("n1", exact=True)

    diff = adds - n1
    assert len(diff) == 1
    assert diff.first().name == "n3"

    union = adds | n1
    assert len(union) == 2

    inter = adds & n1
    assert len(inter) == 1
    assert inter.first().name == "n1"


def test_query_shape_dtype(branching_model):
    parser = ONNXParser(branching_model)
    q = parser.find()

    n2 = q.name("n2")
    assert n2.rank(2).count() == 1
    assert n2.rank(3).count() == 0
    assert n2.dtype("float32").count() == 1
    assert n2.dtype("int64").count() == 0


def test_query_chaining(branching_model):
    parser = ONNXParser(branching_model)
    # All Add nodes whose output rank is 2
    res = parser.find().op("Add").rank(2)
    assert len(res) == 2

    # The first Add node
    assert res.first().name == "n1"


def test_query_indexing(branching_model):
    parser = ONNXParser(branching_model)
    q = parser.find().op("Add")
    assert q[0].first().name == "n1"
    assert q[1].first().name == "n3"
    with pytest.raises(IndexError):
        _ = q[2]
