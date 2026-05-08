import onnx
from onnx import TensorProto, helper

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
node1 = helper.make_node("Relu", ["X"], ["Y"])
node2 = helper.make_node("Relu", ["Y"], ["Z"])

graph_def = helper.make_graph(
    [node1, node2],
    "test-model",
    [X],
    [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])],
)
model_def = helper.make_model(graph_def, producer_name="onnx-example")
try:
    onnx.checker.check_model(model_def)
    print("Model 1 Valid")
except Exception as e:
    print("Model 1 Error:", e)

# Invalid topological order
graph_def2 = helper.make_graph(
    [node2, node1],
    "test-model2",
    [X],
    [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])],
)
model_def2 = helper.make_model(graph_def2, producer_name="onnx-example")
try:
    onnx.checker.check_model(model_def2)
    print("Model 2 Valid")
except Exception as e:
    print("Model 2 Error:", e)
