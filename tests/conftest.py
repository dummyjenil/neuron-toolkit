import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


@pytest.fixture
def simple_model(tmp_path):
    # Create a simple model: (Identity(A) + B) * C

    node_id = helper.make_node("Identity", ["A"], ["A_id"], name="node_id")
    node_add = helper.make_node("Add", ["A_id", "B"], ["add_out"], name="node_add")
    node_mul = helper.make_node("Mul", ["add_out", "C"], ["output"], name="node_mul")

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 3])
    B_val = np.array([[1, 2, 3]], dtype=np.float32)
    B = helper.make_tensor("B", TensorProto.FLOAT, [1, 3], B_val.tobytes(), raw=True)
    C_val = np.array([[2, 2, 2]], dtype=np.float32)
    C = helper.make_tensor("C", TensorProto.FLOAT, [1, 3], C_val.tobytes(), raw=True)

    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

    graph = helper.make_graph([node_id, node_add, node_mul], "test_graph", [A], [output], [B, C])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    model_path = str(tmp_path / "simple.onnx")
    onnx.save(model, model_path)
    return model_path


@pytest.fixture
def complex_model(tmp_path):
    # A -> Identity -> Relu -> Add(B) -> Mul(C) -> Sigmoid -> Out
    #                         |
    #                         -> Tanh -> Out2

    node0 = helper.make_node("Identity", ["A"], ["A_id"], name="n0")
    node1 = helper.make_node("Relu", ["A_id"], ["relu_out"], name="n1")
    node2 = helper.make_node("Add", ["relu_out", "B"], ["add_out"], name="n2")
    node3 = helper.make_node("Mul", ["add_out", "C"], ["mul_out"], name="n3")
    node4 = helper.make_node("Sigmoid", ["mul_out"], ["output"], name="n4")
    node5 = helper.make_node("Tanh", ["add_out"], ["output2"], name="n5")

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 10])
    B = helper.make_tensor(
        "B",
        TensorProto.FLOAT,
        [1, 10],
        np.random.randn(1, 10).astype(np.float32).tobytes(),
        raw=True,
    )
    C = helper.make_tensor(
        "C",
        TensorProto.FLOAT,
        [1, 10],
        np.random.randn(1, 10).astype(np.float32).tobytes(),
        raw=True,
    )

    out1 = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])
    out2 = helper.make_tensor_value_info("output2", TensorProto.FLOAT, [1, 10])

    graph = helper.make_graph(
        [node0, node1, node2, node3, node4, node5], "complex_graph", [A], [out1, out2], [B, C]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model_path = str(tmp_path / "complex.onnx")
    onnx.save(model, model_path)
    return model_path


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
        [node0, node1, node2, node3], "branching", [A], [out1, out2], [B, C, D]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = str(tmp_path / "branching.onnx")
    onnx.save(model, path)
    return path
