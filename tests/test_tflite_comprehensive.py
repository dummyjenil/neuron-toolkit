import os

import flatbuffers
import numpy as np
import pytest
import tflite

from neuron_toolkit import NeuronGraph


def create_complex_tflite_model(path):
    """Create a TFLite model: A -> ADD(B) -> add_out -> SUB(C) -> sub_out -> MUL(D) -> output."""
    builder = flatbuffers.Builder(2048)

    # Tensor Names
    t0_name = builder.CreateString("A")
    t1_name = builder.CreateString("B")
    t2_name = builder.CreateString("add_out")
    t3_name = builder.CreateString("C")
    t4_name = builder.CreateString("sub_out")
    t5_name = builder.CreateString("D")
    t6_name = builder.CreateString("output")

    # Shapes
    tflite.TensorStartShapeVector(builder, 1)
    builder.PrependInt32(3)
    shape_vec = builder.EndVector()

    # Create Tensors
    # Tensor 0: A
    tflite.TensorStart(builder)
    tflite.TensorAddName(builder, t0_name)
    tflite.TensorAddShape(builder, shape_vec)
    tflite.TensorAddType(builder, tflite.TensorType.FLOAT32)
    tflite.TensorAddBuffer(builder, 0)
    t0 = tflite.TensorEnd(builder)

    # Tensor 1: B
    tflite.TensorStart(builder)
    tflite.TensorAddName(builder, t1_name)
    tflite.TensorAddShape(builder, shape_vec)
    tflite.TensorAddType(builder, tflite.TensorType.FLOAT32)
    tflite.TensorAddBuffer(builder, 1)
    t1 = tflite.TensorEnd(builder)

    # Tensor 2: add_out
    tflite.TensorStart(builder)
    tflite.TensorAddName(builder, t2_name)
    tflite.TensorAddShape(builder, shape_vec)
    tflite.TensorAddType(builder, tflite.TensorType.FLOAT32)
    tflite.TensorAddBuffer(builder, 0)
    t2 = tflite.TensorEnd(builder)

    # Tensor 3: C
    tflite.TensorStart(builder)
    tflite.TensorAddName(builder, t3_name)
    tflite.TensorAddShape(builder, shape_vec)
    tflite.TensorAddType(builder, tflite.TensorType.FLOAT32)
    tflite.TensorAddBuffer(builder, 2)
    t3 = tflite.TensorEnd(builder)

    # Tensor 4: sub_out
    tflite.TensorStart(builder)
    tflite.TensorAddName(builder, t4_name)
    tflite.TensorAddShape(builder, shape_vec)
    tflite.TensorAddType(builder, tflite.TensorType.FLOAT32)
    tflite.TensorAddBuffer(builder, 0)
    t4 = tflite.TensorEnd(builder)

    # Tensor 5: D
    tflite.TensorStart(builder)
    tflite.TensorAddName(builder, t5_name)
    tflite.TensorAddShape(builder, shape_vec)
    tflite.TensorAddType(builder, tflite.TensorType.FLOAT32)
    tflite.TensorAddBuffer(builder, 3)
    t5 = tflite.TensorEnd(builder)

    # Tensor 6: output
    tflite.TensorStart(builder)
    tflite.TensorAddName(builder, t6_name)
    tflite.TensorAddShape(builder, shape_vec)
    tflite.TensorAddType(builder, tflite.TensorType.FLOAT32)
    tflite.TensorAddBuffer(builder, 0)
    t6 = tflite.TensorEnd(builder)

    # Subgraph Tensors Vector
    tflite.SubGraphStartTensorsVector(builder, 7)
    builder.PrependUOffsetTRelative(t6)
    builder.PrependUOffsetTRelative(t5)
    builder.PrependUOffsetTRelative(t4)
    builder.PrependUOffsetTRelative(t3)
    builder.PrependUOffsetTRelative(t2)
    builder.PrependUOffsetTRelative(t1)
    builder.PrependUOffsetTRelative(t0)
    tensors_vec = builder.EndVector()

    # Subgraph Inputs (tensor 0 is A)
    tflite.SubGraphStartInputsVector(builder, 1)
    builder.PrependInt32(0)
    inputs_vec = builder.EndVector()

    # Subgraph Outputs (tensor 6 is output)
    tflite.SubGraphStartOutputsVector(builder, 1)
    builder.PrependInt32(6)
    outputs_vec = builder.EndVector()

    # Operator 0: ADD (opcode 0, inputs: [0, 1], outputs: [2])
    tflite.OperatorStartInputsVector(builder, 2)
    builder.PrependInt32(1)
    builder.PrependInt32(0)
    op0_in = builder.EndVector()

    tflite.OperatorStartOutputsVector(builder, 1)
    builder.PrependInt32(2)
    op0_out = builder.EndVector()

    tflite.OperatorStart(builder)
    tflite.OperatorAddOpcodeIndex(builder, 0)
    tflite.OperatorAddInputs(builder, op0_in)
    tflite.OperatorAddOutputs(builder, op0_out)
    op0 = tflite.OperatorEnd(builder)

    # Operator 1: SUB (opcode 1, inputs: [2, 3], outputs: [4])
    tflite.OperatorStartInputsVector(builder, 2)
    builder.PrependInt32(3)
    builder.PrependInt32(2)
    op1_in = builder.EndVector()

    tflite.OperatorStartOutputsVector(builder, 1)
    builder.PrependInt32(4)
    op1_out = builder.EndVector()

    tflite.OperatorStart(builder)
    tflite.OperatorAddOpcodeIndex(builder, 1)
    tflite.OperatorAddInputs(builder, op1_in)
    tflite.OperatorAddOutputs(builder, op1_out)
    op1 = tflite.OperatorEnd(builder)

    # Operator 2: MUL (opcode 2, inputs: [4, 5], outputs: [6])
    tflite.OperatorStartInputsVector(builder, 2)
    builder.PrependInt32(5)
    builder.PrependInt32(4)
    op2_in = builder.EndVector()

    tflite.OperatorStartOutputsVector(builder, 1)
    builder.PrependInt32(6)
    op2_out = builder.EndVector()

    tflite.OperatorStart(builder)
    tflite.OperatorAddOpcodeIndex(builder, 2)
    tflite.OperatorAddInputs(builder, op2_in)
    tflite.OperatorAddOutputs(builder, op2_out)
    op2 = tflite.OperatorEnd(builder)

    tflite.SubGraphStartOperatorsVector(builder, 3)
    builder.PrependUOffsetTRelative(op2)
    builder.PrependUOffsetTRelative(op1)
    builder.PrependUOffsetTRelative(op0)
    ops_vec = builder.EndVector()

    # Subgraph Start/End
    tflite.SubGraphStart(builder)
    tflite.SubGraphAddTensors(builder, tensors_vec)
    tflite.SubGraphAddInputs(builder, inputs_vec)
    tflite.SubGraphAddOutputs(builder, outputs_vec)
    tflite.SubGraphAddOperators(builder, ops_vec)
    subgraph = tflite.SubGraphEnd(builder)

    tflite.ModelStartSubgraphsVector(builder, 1)
    builder.PrependUOffsetTRelative(subgraph)
    subgraphs_vec = builder.EndVector()

    # Opcodes
    tflite.OperatorCodeStart(builder)
    tflite.OperatorCodeAddDeprecatedBuiltinCode(
        builder, min(tflite.BuiltinOperator.ADD, 127)
    )
    tflite.OperatorCodeAddBuiltinCode(builder, tflite.BuiltinOperator.ADD)
    opcode0 = tflite.OperatorCodeEnd(builder)

    tflite.OperatorCodeStart(builder)
    tflite.OperatorCodeAddDeprecatedBuiltinCode(
        builder, min(tflite.BuiltinOperator.SUB, 127)
    )
    tflite.OperatorCodeAddBuiltinCode(builder, tflite.BuiltinOperator.SUB)
    opcode1 = tflite.OperatorCodeEnd(builder)

    tflite.OperatorCodeStart(builder)
    tflite.OperatorCodeAddDeprecatedBuiltinCode(
        builder, min(tflite.BuiltinOperator.MUL, 127)
    )
    tflite.OperatorCodeAddBuiltinCode(builder, tflite.BuiltinOperator.MUL)
    opcode2 = tflite.OperatorCodeEnd(builder)

    tflite.ModelStartOperatorCodesVector(builder, 3)
    builder.PrependUOffsetTRelative(opcode2)
    builder.PrependUOffsetTRelative(opcode1)
    builder.PrependUOffsetTRelative(opcode0)
    opcodes_vec = builder.EndVector()

    # Buffers
    # Buffer 0: Empty
    tflite.BufferStart(builder)
    b0 = tflite.BufferEnd(builder)

    # Buffer 1: B data ([1.0, 2.0, 3.0])
    b1_data = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
    b1_vec = builder.CreateByteVector(b1_data)
    tflite.BufferStart(builder)
    tflite.BufferAddData(builder, b1_vec)
    b1 = tflite.BufferEnd(builder)

    # Buffer 2: C data ([4.0, 5.0, 6.0])
    b2_data = np.array([4.0, 5.0, 6.0], dtype=np.float32).tobytes()
    b2_vec = builder.CreateByteVector(b2_data)
    tflite.BufferStart(builder)
    tflite.BufferAddData(builder, b2_vec)
    b2 = tflite.BufferEnd(builder)

    # Buffer 3: D data ([7.0, 8.0, 9.0])
    b3_data = np.array([7.0, 8.0, 9.0], dtype=np.float32).tobytes()
    b3_vec = builder.CreateByteVector(b3_data)
    tflite.BufferStart(builder)
    tflite.BufferAddData(builder, b3_vec)
    b3 = tflite.BufferEnd(builder)

    tflite.ModelStartBuffersVector(builder, 4)
    builder.PrependUOffsetTRelative(b3)
    builder.PrependUOffsetTRelative(b2)
    builder.PrependUOffsetTRelative(b1)
    builder.PrependUOffsetTRelative(b0)
    buffers_vec = builder.EndVector()

    # Model
    tflite.ModelStart(builder)
    tflite.ModelAddSubgraphs(builder, subgraphs_vec)
    tflite.ModelAddOperatorCodes(builder, opcodes_vec)
    tflite.ModelAddBuffers(builder, buffers_vec)
    model = tflite.ModelEnd(builder)

    builder.Finish(model)
    with open(path, "wb") as f:
        f.write(builder.Output())


@pytest.fixture
def complex_tflite_path(tmp_path):
    path = os.path.join(tmp_path, "complex.tflite")
    create_complex_tflite_model(path)
    return path


def test_tflite_graph_properties(complex_tflite_path):
    """Test basic parser loading and graph property retrieval on TFLite."""
    g = NeuronGraph(complex_tflite_path)

    # Nodes
    assert len(g.nodes) == 3
    assert [n.op_type for n in g.nodes] == ["ADD", "SUB", "MUL"]

    # Initializers (tensor_map)
    assert "B" in g.tensor_map
    assert "C" in g.tensor_map
    assert "D" in g.tensor_map
    assert "A" not in g.tensor_map
    assert "add_out" not in g.tensor_map
    assert "output" not in g.tensor_map

    # Shape Info
    assert g.shape_info["A"] == (1, "float32")
    assert g.shape_info["B"] == (1, "float32")
    assert g.shape_info["add_out"] == (1, "float32")
    assert g.shape_info["output"] == (1, "float32")


def test_tflite_query_filters(complex_tflite_path):
    """Test query engine filters on TFLite."""
    g = NeuronGraph(complex_tflite_path)
    q = g.query()

    # op_type filters
    assert q.op("ADD").count() == 1
    assert q.op("SUB").count() == 1
    assert q.op("MUL").count() == 1

    # name filters
    assert q.name("ADD_0", exact=True).count() == 1
    assert q.name("SUB_1", exact=True).count() == 1

    # rank & dtype filters
    assert q.rank(1).count() == 3
    assert q.rank(2).count() == 0
    assert q.dtype("float32").count() == 3
    assert q.dtype("int32").count() == 0


def test_tflite_traversal(complex_tflite_path):
    """Test graph traversal on TFLite backend."""
    g = NeuronGraph(complex_tflite_path)
    q = g.query()

    add_node = q.op("ADD")
    sub_node = q.op("SUB")
    mul_node = q.op("MUL")

    # Direct children / outputs
    assert add_node.children().single_node.op_type == "SUB"
    assert sub_node.outputs().single_node.op_type == "MUL"

    # Direct parents / inputs
    assert sub_node.parents().single_node.op_type == "ADD"
    assert mul_node.inputs().single_node.op_type == "SUB"

    # Descendants & Ancestors
    assert {n.op_type for n in add_node.descendants()} == {"SUB", "MUL"}
    assert {n.op_type for n in mul_node.ancestors()} == {"ADD", "SUB"}

    # Entry & Output Nodes
    assert q.entry_nodes.single_node.op_type == "ADD"
    assert q.output_nodes.single_node.op_type == "MUL"


def test_tflite_set_operations(complex_tflite_path):
    """Test set operations (union, intersection, difference) on TFLite query objects."""
    g = NeuronGraph(complex_tflite_path)
    q = g.query()

    q_add = q.op("ADD")
    q_sub = q.op("SUB")

    # Union
    union = q_add | q_sub
    assert union.count() == 2
    assert {n.op_type for n in union} == {"ADD", "SUB"}

    # Intersection
    inter = union & q_add
    assert inter.count() == 1
    assert inter.single_node.op_type == "ADD"

    # Difference
    diff = union - q_add
    assert diff.count() == 1
    assert diff.single_node.op_type == "SUB"


def test_tflite_parameter_extraction(complex_tflite_path):
    """Test parameter extraction filters and values on TFLite."""
    g = NeuronGraph(complex_tflite_path)
    q = g.query()

    # has_params() filter
    assert q.has_params().count() == 3  # All nodes have constant weight inputs B, C, D

    # Get weight tensor values
    add_weights = q.op("ADD").tensor()
    assert isinstance(add_weights, dict)
    assert "B" in add_weights
    assert "A" not in add_weights  # Dynamic input should not be in params
    assert np.allclose(add_weights["B"], [1.0, 2.0, 3.0])

    sub_weights = q.op("SUB").tensor()
    assert "C" in sub_weights
    assert np.allclose(sub_weights["C"], [4.0, 5.0, 6.0])

    # single_tensor accessor
    assert np.allclose(q.op("MUL").single_tensor, [7.0, 8.0, 9.0])


def test_tflite_topological_sort(complex_tflite_path):
    """Test topological sorting on TFLite."""
    g = NeuronGraph(complex_tflite_path)
    q = g.query()

    # Shuffle nodes manually to verify sorting works
    nodes = list(reversed(q.nodes))
    shuffled_q = q._clone(nodes)
    assert not shuffled_q.is_topologically_sorted()

    sorted_q = shuffled_q.topological_sort()
    assert sorted_q.is_topologically_sorted()
    assert [n.op_type for n in sorted_q] == ["ADD", "SUB", "MUL"]
