import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from onnx_toolkit.parser import ONNXParser
from onnx_toolkit.pattern import Pattern


@pytest.fixture
def sub_model(tmp_path):
    # node_sub = Sub(Identity(A), B)  where B=1.0
    node_id = helper.make_node("Identity", ["A"], ["A_id"], name="node_id")
    node_sub = helper.make_node("Sub", ["A_id", "B"], ["output"], name="node_sub")

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1])
    B_val = np.array([1.0], dtype=np.float32)
    B = helper.make_tensor("B", TensorProto.FLOAT, [1], B_val.tobytes(), raw=True)

    graph = helper.make_graph([node_id, node_sub], "graph", [A], [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])], [B])
    model = helper.make_model(graph)
    path = str(tmp_path / "sub.onnx")
    onnx.save(model, path)
    return path

def test_pattern_sub_order(sub_model):
    parser = ONNXParser(sub_model)

    # Correct order: Sub(any, 1.0)
    pat_correct = Pattern.op("Sub", Pattern.any(), Pattern.const(1.0))
    assert parser.pattern_detect(pat_correct, start_node="node_sub") is not None

    # Wrong order: Sub(1.0, any)
    # This should NOT match Sub(A_id, 1.0)
    pat_wrong = Pattern.op("Sub", Pattern.const(1.0), Pattern.any())
    res = parser.pattern_detect(pat_wrong, start_node="node_sub")

    # If the bug exists, this will match
    assert res is None, "Should not match Sub(A_id, 1.0) with pattern Sub(1.0, any)"
