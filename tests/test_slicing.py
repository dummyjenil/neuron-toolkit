import os

import flatbuffers
import numpy as np
import pytest
import tflite
import torch
from torch import nn

from neuron_toolkit.graph import NeuronGraph


def create_multi_node_tflite_model(path):
    """Create a 3-node TFLite model: input -> ABS -> abs_out -> ADD (with const weight) -> add_out -> SUB -> output."""
    builder = flatbuffers.Builder(2048)

    # Tensor Names
    t_input = builder.CreateString("input")
    t_abs_out = builder.CreateString("abs_out")
    t_const = builder.CreateString("const_w")
    t_add_out = builder.CreateString("add_out")
    t_output = builder.CreateString("output")

    # Shapes
    tflite.TensorStartShapeVector(builder, 2)
    builder.PrependInt32(10)
    builder.PrependInt32(1)
    shape_vec = builder.EndVector()

    # Tensors Serialization
    # input (idx 0)
    tflite.TensorStart(builder)
    tflite.TensorAddName(builder, t_input)
    tflite.TensorAddShape(builder, shape_vec)
    tflite.TensorAddType(builder, tflite.TensorType.FLOAT32)
    t1 = tflite.TensorEnd(builder)

    # abs_out (idx 1)
    tflite.TensorStart(builder)
    tflite.TensorAddName(builder, t_abs_out)
    tflite.TensorAddShape(builder, shape_vec)
    tflite.TensorAddType(builder, tflite.TensorType.FLOAT32)
    t2 = tflite.TensorEnd(builder)

    # const_w (idx 2) - Buffer idx 1
    tflite.TensorStart(builder)
    tflite.TensorAddName(builder, t_const)
    tflite.TensorAddShape(builder, shape_vec)
    tflite.TensorAddType(builder, tflite.TensorType.FLOAT32)
    tflite.TensorAddBuffer(builder, 1)
    t3 = tflite.TensorEnd(builder)

    # add_out (idx 3)
    tflite.TensorStart(builder)
    tflite.TensorAddName(builder, t_add_out)
    tflite.TensorAddShape(builder, shape_vec)
    tflite.TensorAddType(builder, tflite.TensorType.FLOAT32)
    t4 = tflite.TensorEnd(builder)

    # output (idx 4)
    tflite.TensorStart(builder)
    tflite.TensorAddName(builder, t_output)
    tflite.TensorAddShape(builder, shape_vec)
    tflite.TensorAddType(builder, tflite.TensorType.FLOAT32)
    t5 = tflite.TensorEnd(builder)

    # Subgraph Tensors Vector
    tflite.SubGraphStartTensorsVector(builder, 5)
    builder.PrependUOffsetTRelative(t5)
    builder.PrependUOffsetTRelative(t4)
    builder.PrependUOffsetTRelative(t3)
    builder.PrependUOffsetTRelative(t2)
    builder.PrependUOffsetTRelative(t1)
    tensors_vec = builder.EndVector()

    # Inputs/Outputs
    tflite.SubGraphStartInputsVector(builder, 1)
    builder.PrependInt32(0)
    inputs_vec = builder.EndVector()

    tflite.SubGraphStartOutputsVector(builder, 1)
    builder.PrependInt32(4)
    outputs_vec = builder.EndVector()

    # Operator 1: ABS (inputs: [0], outputs: [1])
    tflite.OperatorStartInputsVector(builder, 1)
    builder.PrependInt32(0)
    op1_in = builder.EndVector()

    tflite.OperatorStartOutputsVector(builder, 1)
    builder.PrependInt32(1)
    op1_out = builder.EndVector()

    tflite.OperatorStart(builder)
    tflite.OperatorAddOpcodeIndex(builder, 0)
    tflite.OperatorAddInputs(builder, op1_in)
    tflite.OperatorAddOutputs(builder, op1_out)
    op1 = tflite.OperatorEnd(builder)

    # Operator 2: ADD (inputs: [1, 2], outputs: [3], options: fused ReLU)
    tflite.OperatorStartInputsVector(builder, 2)
    builder.PrependInt32(2)
    builder.PrependInt32(1)
    op2_in = builder.EndVector()

    tflite.OperatorStartOutputsVector(builder, 1)
    builder.PrependInt32(3)
    op2_out = builder.EndVector()

    tflite.AddOptionsStart(builder)
    tflite.AddOptionsAddFusedActivationFunction(
        builder, tflite.ActivationFunctionType.RELU
    )
    add_opts = tflite.AddOptionsEnd(builder)

    tflite.OperatorStart(builder)
    tflite.OperatorAddOpcodeIndex(builder, 1)
    tflite.OperatorAddInputs(builder, op2_in)
    tflite.OperatorAddOutputs(builder, op2_out)
    tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.AddOptions)
    tflite.OperatorAddBuiltinOptions(builder, add_opts)
    op2 = tflite.OperatorEnd(builder)

    # Operator 3: SUB (inputs: [3, 0], outputs: [4])
    tflite.OperatorStartInputsVector(builder, 2)
    builder.PrependInt32(0)
    builder.PrependInt32(3)
    op3_in = builder.EndVector()

    tflite.OperatorStartOutputsVector(builder, 1)
    builder.PrependInt32(4)
    op3_out = builder.EndVector()

    tflite.OperatorStart(builder)
    tflite.OperatorAddOpcodeIndex(builder, 2)
    tflite.OperatorAddInputs(builder, op3_in)
    tflite.OperatorAddOutputs(builder, op3_out)
    op3 = tflite.OperatorEnd(builder)

    # Operators Subgraph Vector
    tflite.SubGraphStartOperatorsVector(builder, 3)
    builder.PrependUOffsetTRelative(op3)
    builder.PrependUOffsetTRelative(op2)
    builder.PrependUOffsetTRelative(op1)
    ops_vec = builder.EndVector()

    # Subgraph
    tflite.SubGraphStart(builder)
    tflite.SubGraphAddTensors(builder, tensors_vec)
    tflite.SubGraphAddInputs(builder, inputs_vec)
    tflite.SubGraphAddOutputs(builder, outputs_vec)
    tflite.SubGraphAddOperators(builder, ops_vec)
    subgraph = tflite.SubGraphEnd(builder)

    tflite.ModelStartSubgraphsVector(builder, 1)
    builder.PrependUOffsetTRelative(subgraph)
    subgraphs_vec = builder.EndVector()

    # Opcodes (ABS, ADD, SUB)
    tflite.OperatorCodeStart(builder)
    tflite.OperatorCodeAddDeprecatedBuiltinCode(builder, tflite.BuiltinOperator.ABS)
    tflite.OperatorCodeAddBuiltinCode(builder, tflite.BuiltinOperator.ABS)
    op_abs = tflite.OperatorCodeEnd(builder)

    tflite.OperatorCodeStart(builder)
    tflite.OperatorCodeAddDeprecatedBuiltinCode(builder, tflite.BuiltinOperator.ADD)
    tflite.OperatorCodeAddBuiltinCode(builder, tflite.BuiltinOperator.ADD)
    op_add = tflite.OperatorCodeEnd(builder)

    tflite.OperatorCodeStart(builder)
    tflite.OperatorCodeAddDeprecatedBuiltinCode(builder, tflite.BuiltinOperator.SUB)
    tflite.OperatorCodeAddBuiltinCode(builder, tflite.BuiltinOperator.SUB)
    op_sub = tflite.OperatorCodeEnd(builder)

    tflite.ModelStartOperatorCodesVector(builder, 3)
    builder.PrependUOffsetTRelative(op_sub)
    builder.PrependUOffsetTRelative(op_add)
    builder.PrependUOffsetTRelative(op_abs)
    opcodes_vec = builder.EndVector()

    # Buffers
    tflite.BufferStart(builder)
    b0 = tflite.BufferEnd(builder)

    const_w_data = np.ones((1, 10), dtype=np.float32).tobytes()
    b1_data = builder.CreateByteVector(const_w_data)
    tflite.BufferStart(builder)
    tflite.BufferAddData(builder, b1_data)
    b1 = tflite.BufferEnd(builder)

    tflite.ModelStartBuffersVector(builder, 2)
    builder.PrependUOffsetTRelative(b1)
    builder.PrependUOffsetTRelative(b0)
    buffers_vec = builder.EndVector()

    # Model
    tflite.ModelStart(builder)
    tflite.ModelAddVersion(builder, 3)
    tflite.ModelAddSubgraphs(builder, subgraphs_vec)
    tflite.ModelAddOperatorCodes(builder, opcodes_vec)
    tflite.ModelAddBuffers(builder, buffers_vec)
    model = tflite.ModelEnd(builder)

    builder.Finish(model, file_identifier=b"TFL3")
    with open(path, "wb") as f:
        f.write(builder.Output())


@pytest.fixture
def complex_tflite_model(tmp_path):
    path = os.path.join(tmp_path, "complex.tflite")
    create_multi_node_tflite_model(path)
    return path


def test_onnx_slicing(complex_model, tmp_path):
    """Test dynamic slicing on an ONNX model."""
    graph = NeuronGraph(complex_model)

    # Original model has 6 nodes: Identity -> Relu -> Add -> Mul -> Sigmoid & Tanh
    assert len(graph.nodes) == 6

    # Slice between relu_out and mul_out (should keep Add and Mul, plus B and C parameters)
    sliced = graph.slice("relu_out", "mul_out")
    assert len(sliced.nodes) == 2
    op_types = {n.op_type for n in sliced.nodes}
    assert op_types == {"Add", "Mul"}

    # Verify input/output names
    assert "relu_out" in sliced._backend.graph_inputs
    assert "mul_out" in sliced._backend.graph_outputs

    # Verify we can save it
    out_path = str(tmp_path / "onnx_sliced.onnx")
    sliced.save(out_path)
    assert os.path.exists(out_path)

    # Reload and verify
    reloaded = NeuronGraph(out_path)
    assert len(reloaded.nodes) == 2


def test_onnx_single_node_extraction(complex_model, tmp_path):
    """Test single node extraction on an ONNX model."""
    graph = NeuronGraph(complex_model)

    # Find the Relu node
    relu_node = next(n for n in graph.nodes if getattr(n, "op_type", "") == "Relu")

    # Extract this node
    sliced = graph.slice_node(relu_node)
    assert len(sliced.nodes) == 1
    assert sliced.nodes[0].op_type == "Relu"

    out_path = str(tmp_path / "onnx_relu.onnx")
    sliced.save(out_path)
    assert os.path.exists(out_path)


def test_tflite_slicing(complex_tflite_model, tmp_path):
    """Test dynamic slicing on a TFLite model."""
    graph = NeuronGraph(complex_tflite_model)
    assert len(graph.nodes) == 3

    # Slice from abs_out to add_out. Should keep only the ADD node and its parameter const_w
    sliced = graph.slice("abs_out", "add_out")
    assert len(sliced.nodes) == 1
    assert sliced.nodes[0].op_type == "ADD"

    # Verify boundary inputs/outputs
    assert "abs_out" in sliced._backend.graph_inputs
    assert "add_out" in sliced._backend.graph_outputs
    assert "const_w" in sliced.tensor_map

    # Verify save & reload
    out_path = str(tmp_path / "tflite_sliced.tflite")
    sliced.save(out_path)
    assert os.path.exists(out_path)

    reloaded = NeuronGraph(out_path)
    assert len(reloaded.nodes) == 1
    assert reloaded.nodes[0].op_type == "ADD"
    assert "const_w" in reloaded.tensor_map


def test_tflite_single_node_extraction(complex_tflite_model, tmp_path):
    """Test single node extraction on a TFLite model."""
    graph = NeuronGraph(complex_tflite_model)

    # Extract Operator 0 (ABS)
    abs_node = graph.nodes[0]
    sliced = graph.slice_node(abs_node)
    assert len(sliced.nodes) == 1
    assert sliced.nodes[0].op_type == "ABS"

    out_path = str(tmp_path / "tflite_abs.tflite")
    sliced.save(out_path)
    assert os.path.exists(out_path)

    reloaded = NeuronGraph(out_path)
    assert len(reloaded.nodes) == 1
    assert reloaded.nodes[0].op_type == "ABS"


def test_onnx_pytorch_comparison(complex_model):
    """Test verification of ONNX model slice outputs against PyTorch."""
    graph = NeuronGraph(complex_model)

    # Extract weights B and C from graph
    weights = graph.get_weights()
    B_np = weights["B"]
    C_np = weights["C"]

    # Implement matching PyTorch model
    class PyTorchReluAddMul(nn.Module):
        def __init__(self, B, C):
            super().__init__()
            self.B = torch.tensor(B)
            self.C = torch.tensor(C)

        def forward(self, x):
            return (torch.relu(x) + self.B) * self.C

    model_py = PyTorchReluAddMul(B_np, C_np)
    sample_input = {"A_id": np.random.randn(1, 10).astype(np.float32)}

    # Verify comparison
    report = graph.compare_outputs(
        start_points="A_id",
        end_points="mul_out",
        pytorch_module=model_py,
        inputs_dict=sample_input,
    )

    assert "mul_out" in report or "output_0" in report
    key = "mul_out" if "mul_out" in report else "output_0"
    assert report[key]["all_close"] is True
    assert report[key]["cosine_similarity"] > 0.999
    assert report[key]["mae"] < 1e-5


def test_pytorch_comparison(complex_tflite_model):
    """Test verification against PyTorch."""
    from neuron_toolkit.graph import _is_tflite_runnable

    if not _is_tflite_runnable():
        pytest.skip("LiteRT interpreter not runnable on this CPU.")

    graph = NeuronGraph(complex_tflite_model)

    # Slice ABS layer (from input to abs_out)
    # TFLite model: input -> ABS -> abs_out

    # Implement ABS in PyTorch
    class PyTorchAbs(nn.Module):
        def forward(self, x):
            return torch.abs(x)

    model_py = PyTorchAbs()
    sample_input = {"input": np.random.randn(1, 10).astype(np.float32)}

    # Verify comparison outputs
    report = graph.compare_outputs(
        start_points="input",
        end_points="abs_out",
        pytorch_module=model_py,
        inputs_dict=sample_input,
    )

    # Check that report is computed and similarity is extremely high
    assert "abs_out" in report or "output_0" in report
    key = "abs_out" if "abs_out" in report else "output_0"
    assert report[key]["all_close"] is True
    assert report[key]["cosine_similarity"] > 0.999
    assert report[key]["mae"] < 1e-6

    # Test Decorator verification
    @graph.verify_against_pytorch(start_points="input", end_points="abs_out")
    def run_layer(x):
        return torch.abs(x)

    tensor_input = torch.from_numpy(sample_input["input"])
    _ = run_layer(tensor_input)

    assert hasattr(run_layer, "similarity_report")
    rep_dec = run_layer.similarity_report
    assert key in rep_dec
    assert rep_dec[key]["all_close"] is True
