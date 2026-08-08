import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


def build_tflite_model(
    codes,
    tensors,
    ops,
    inputs,
    outputs,
    buffers,
    identifier=b"TFL3",
):
    """Build a TFLite flatbuffer model (with real ``TFL3`` file identifier).

    Args:
        codes: list of builtin opcodes for the OperatorCodes table.
        tensors: list of dicts {"name", "shape", "dtype", "buffer"}.
        ops: list of dicts {"index": opcode_index, "inputs": [..], "outputs": [..]}.
        inputs/outputs: subgraph input/output tensor indices.
        buffers: list of raw ``bytes`` (or ``None``). buffers[0] must be empty.
        identifier: file identifier; ``b"TFL3"`` mimics real TF Lite files.
    """
    import flatbuffers
    import tflite

    builder = flatbuffers.Builder(2048)

    tensor_offsets = []
    for t in tensors:
        name = builder.CreateString(t["name"])
        tflite.TensorStartShapeVector(builder, len(t["shape"]))
        for s in reversed(t["shape"]):
            builder.PrependInt32(s)
        shape_vec = builder.EndVector()

        tflite.TensorStart(builder)
        tflite.TensorAddName(builder, name)
        tflite.TensorAddShape(builder, shape_vec)
        tflite.TensorAddType(builder, t["dtype"])
        tflite.TensorAddBuffer(builder, t["buffer"])
        tensor_offsets.append(tflite.TensorEnd(builder))

    tflite.SubGraphStartTensorsVector(builder, len(tensor_offsets))
    for off in reversed(tensor_offsets):
        builder.PrependUOffsetTRelative(off)
    tensors_vec = builder.EndVector()

    op_offsets = []
    for op in ops:
        tflite.OperatorStartInputsVector(builder, len(op["inputs"]))
        for x in reversed(op["inputs"]):
            builder.PrependInt32(x)
        inputs_vec = builder.EndVector()

        tflite.OperatorStartOutputsVector(builder, len(op["outputs"]))
        for x in reversed(op["outputs"]):
            builder.PrependInt32(x)
        outputs_vec = builder.EndVector()

        builtin_options_type = op.get("builtin_options_type", 0)
        builtin_options = op.get("builtin_options", 0)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, op["index"])
        tflite.OperatorAddInputs(builder, inputs_vec)
        tflite.OperatorAddOutputs(builder, outputs_vec)
        if builtin_options_type:
            tflite.OperatorAddBuiltinOptionsType(builder, builtin_options_type)
        if builtin_options:
            tflite.OperatorAddBuiltinOptions(builder, builtin_options)
        op_offsets.append(tflite.OperatorEnd(builder))

    tflite.SubGraphStartOperatorsVector(builder, len(op_offsets))
    for off in reversed(op_offsets):
        builder.PrependUOffsetTRelative(off)
    ops_vec = builder.EndVector()

    tflite.SubGraphStartInputsVector(builder, len(inputs))
    for i in reversed(inputs):
        builder.PrependInt32(i)
    inputs_vec = builder.EndVector()

    tflite.SubGraphStartOutputsVector(builder, len(outputs))
    for o in reversed(outputs):
        builder.PrependInt32(o)
    outputs_vec = builder.EndVector()

    tflite.SubGraphStart(builder)
    tflite.SubGraphAddTensors(builder, tensors_vec)
    tflite.SubGraphAddInputs(builder, inputs_vec)
    tflite.SubGraphAddOutputs(builder, outputs_vec)
    tflite.SubGraphAddOperators(builder, ops_vec)
    subgraph = tflite.SubGraphEnd(builder)

    tflite.ModelStartSubgraphsVector(builder, 1)
    builder.PrependUOffsetTRelative(subgraph)
    subgraphs_vec = builder.EndVector()

    opcode_offsets = []
    for code in codes:
        tflite.OperatorCodeStart(builder)
        tflite.OperatorCodeAddDeprecatedBuiltinCode(builder, min(code, 127))
        tflite.OperatorCodeAddBuiltinCode(builder, code)
        opcode_offsets.append(tflite.OperatorCodeEnd(builder))

    tflite.ModelStartOperatorCodesVector(builder, len(opcode_offsets))
    for off in reversed(opcode_offsets):
        builder.PrependUOffsetTRelative(off)
    opcodes_vec = builder.EndVector()

    buffer_offsets = []
    for data in buffers:
        data_vec = 0
        if data:
            data_vec = builder.CreateByteVector(data)
        tflite.BufferStart(builder)
        if data_vec:
            tflite.BufferAddData(builder, data_vec)
        buffer_offsets.append(tflite.BufferEnd(builder))

    tflite.ModelStartBuffersVector(builder, len(buffer_offsets))
    for off in reversed(buffer_offsets):
        builder.PrependUOffsetTRelative(off)
    buffers_vec = builder.EndVector()

    tflite.ModelStart(builder)
    tflite.ModelAddVersion(builder, 3)
    tflite.ModelAddSubgraphs(builder, subgraphs_vec)
    tflite.ModelAddOperatorCodes(builder, opcodes_vec)
    tflite.ModelAddBuffers(builder, buffers_vec)
    model = tflite.ModelEnd(builder)

    builder.Finish(model, file_identifier=identifier)
    return bytes(builder.Output())


@pytest.fixture
def tflite_abs_bytes():
    """TFLite bytes: input -> ABS -> output (with TFL3 magic)."""
    import tflite

    return build_tflite_model(
        codes=[tflite.BuiltinOperator.ABS],
        tensors=[
            {
                "name": "input",
                "shape": [3],
                "dtype": tflite.TensorType.FLOAT32,
                "buffer": 0,
            },
            {
                "name": "output",
                "shape": [3],
                "dtype": tflite.TensorType.FLOAT32,
                "buffer": 0,
            },
        ],
        ops=[{"index": 0, "inputs": [0], "outputs": [1]}],
        inputs=[0],
        outputs=[1],
        buffers=[None],
    )


@pytest.fixture
def tflite_abs_path(tmp_path, tflite_abs_bytes):
    """Same ABS model written to a file."""
    path = tmp_path / "abs.tflite"
    path.write_bytes(tflite_abs_bytes)
    return str(path)


@pytest.fixture
def tflite_add_const_bytes():
    """TFLite bytes: input -> ADD(const_w=[1,2,3]) -> output (with TFL3 magic)."""
    import tflite

    return build_tflite_model(
        codes=[tflite.BuiltinOperator.ADD],
        tensors=[
            {
                "name": "input",
                "shape": [3],
                "dtype": tflite.TensorType.FLOAT32,
                "buffer": 0,
            },
            {
                "name": "const_w",
                "shape": [3],
                "dtype": tflite.TensorType.FLOAT32,
                "buffer": 1,
            },
            {
                "name": "output",
                "shape": [3],
                "dtype": tflite.TensorType.FLOAT32,
                "buffer": 0,
            },
        ],
        ops=[{"index": 0, "inputs": [0, 1], "outputs": [2]}],
        inputs=[0],
        outputs=[2],
        buffers=[None, np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()],
    )


@pytest.fixture
def tflite_add_const_path(tmp_path, tflite_add_const_bytes):
    """TFLite ADD-const model written to disk."""
    path = tmp_path / "add_const.tflite"
    path.write_bytes(tflite_add_const_bytes)
    return str(path)


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

    graph = helper.make_graph(
        [node_id, node_add, node_mul], "test_graph", [A], [output], [B, C]
    )

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
        [node0, node1, node2, node3, node4, node5],
        "complex_graph",
        [A],
        [out1, out2],
        [B, C],
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
