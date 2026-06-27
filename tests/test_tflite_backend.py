import os

import flatbuffers
import pytest
import tflite

from neuron_toolkit.backends.tflite.parser import TFLiteParser
from neuron_toolkit.pattern import Pattern


def create_minimal_tflite_model(path):
    """Create a minimal TFLite model with one ADD operation."""
    builder = flatbuffers.Builder(1024)

    # Tensors
    t1_name = builder.CreateString("input")
    t2_name = builder.CreateString("output")

    tflite.TensorStartShapeVector(builder, 2)
    builder.PrependInt32(10)
    builder.PrependInt32(1)
    shape_vec = builder.EndVector()

    tflite.TensorStart(builder)
    tflite.TensorAddName(builder, t1_name)
    tflite.TensorAddShape(builder, shape_vec)
    tflite.TensorAddType(builder, tflite.TensorType.FLOAT32)
    t1 = tflite.TensorEnd(builder)

    tflite.TensorStart(builder)
    tflite.TensorAddName(builder, t2_name)
    tflite.TensorAddShape(builder, shape_vec)
    tflite.TensorAddType(builder, tflite.TensorType.FLOAT32)
    t2 = tflite.TensorEnd(builder)

    tflite.SubGraphStartTensorsVector(builder, 2)
    builder.PrependUOffsetTRelative(t2)
    builder.PrependUOffsetTRelative(t1)
    tensors_vec = builder.EndVector()

    # Inputs/Outputs
    tflite.SubGraphStartInputsVector(builder, 1)
    builder.PrependInt32(0)
    inputs_vec = builder.EndVector()

    tflite.SubGraphStartOutputsVector(builder, 1)
    builder.PrependInt32(1)
    outputs_vec = builder.EndVector()

    # Operator
    tflite.OperatorStartInputsVector(builder, 1)
    builder.PrependInt32(0)
    op_in_vec = builder.EndVector()

    tflite.OperatorStartOutputsVector(builder, 1)
    builder.PrependInt32(1)
    op_out_vec = builder.EndVector()

    tflite.OperatorStart(builder)
    tflite.OperatorAddOpcodeIndex(builder, 0)
    tflite.OperatorAddInputs(builder, op_in_vec)
    tflite.OperatorAddOutputs(builder, op_out_vec)
    op = tflite.OperatorEnd(builder)

    tflite.SubGraphStartOperatorsVector(builder, 1)
    builder.PrependUOffsetTRelative(op)
    ops_vec = builder.EndVector()

    # SubGraph
    tflite.SubGraphStart(builder)
    tflite.SubGraphAddTensors(builder, tensors_vec)
    tflite.SubGraphAddInputs(builder, inputs_vec)
    tflite.SubGraphAddOutputs(builder, outputs_vec)
    tflite.SubGraphAddOperators(builder, ops_vec)
    subgraph = tflite.SubGraphEnd(builder)

    tflite.ModelStartSubgraphsVector(builder, 1)
    builder.PrependUOffsetTRelative(subgraph)
    subgraphs_vec = builder.EndVector()

    # Opcode
    tflite.OperatorCodeStart(builder)
    tflite.OperatorCodeAddBuiltinCode(builder, tflite.BuiltinOperator.ADD)
    opcode = tflite.OperatorCodeEnd(builder)

    tflite.ModelStartOperatorCodesVector(builder, 1)
    builder.PrependUOffsetTRelative(opcode)
    opcodes_vec = builder.EndVector()

    # Buffers
    tflite.BufferStart(builder)
    b0 = tflite.BufferEnd(builder)
    tflite.ModelStartBuffersVector(builder, 1)
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
def tflite_model_path(tmp_path):
    path = os.path.join(tmp_path, "test.tflite")
    create_minimal_tflite_model(path)
    return path


def test_tflite_parser_load(tflite_model_path):
    parser = TFLiteParser(tflite_model_path)
    assert len(parser.nodes) == 1
    assert parser.nodes[0].op_type == "ADD"
    assert "input" in parser.nodes[0].input
    assert "output" in parser.nodes[0].output


def test_tflite_query(tflite_model_path):
    parser = TFLiteParser(tflite_model_path)
    q = parser.find().op("ADD")
    assert q.count() == 1


def test_tflite_rewriter_functional(tflite_model_path, tmp_path):
    parser = TFLiteParser(tflite_model_path)
    rw = parser.rewriter()
    node = parser.nodes[0]
    print(f"\nDEBUG: Replacing node {node.name} (op={node.op_type})")

    # Replacement with a "MyFusion" custom op
    out_path = os.path.join(tmp_path, "fused.tflite")
    rw.replace([node], "MyFusion", node.input, node.output)
    print(f"DEBUG: to_remove set: {rw._to_remove}")

    saved_bytes = rw.build(out_path)
    assert os.path.exists(out_path)
    assert isinstance(saved_bytes, bytes)

    # Verify the new model has MyFusion
    new_parser = TFLiteParser(out_path)
    print(
        f"DEBUG: New model nodes: {[n.name for n in new_parser.nodes]} types: {[n.op_type for n in new_parser.nodes]}"
    )
    assert len(new_parser.nodes) == 1
    assert new_parser.nodes[0].op_type == "MyFusion"


def test_tflite_pattern_detect(tflite_model_path):
    parser = TFLiteParser(tflite_model_path)
    pat = Pattern.op("ADD")
    node = parser.nodes[0]
    match = parser.pattern_detect(pat, start_node=node)
    assert match is not None
    assert match.start.op_type == "ADD"


def test_tflite_parser_bytes_and_object(tflite_model_path):
    # Test loading from bytes
    from pathlib import Path

    buf = Path(tflite_model_path).read_bytes()

    parser_bytes = TFLiteParser(buf)
    assert len(parser_bytes.nodes) == 1
    assert parser_bytes.nodes[0].op_type == "ADD"

    # Test loading from tflite.Model object
    import tflite

    model_obj = tflite.Model.GetRootAsModel(buf, 0)
    parser_obj = TFLiteParser(model_obj)
    assert len(parser_obj.nodes) == 1
    assert parser_obj.nodes[0].op_type == "ADD"


def test_tflite_extra_attributes():
    from unittest.mock import MagicMock

    from neuron_toolkit.backends.tflite.utils import _get_tflite_attr

    # If BuiltinOptions returns None, it should return empty dict
    op_mock = MagicMock()
    op_mock.BuiltinOptions.return_value = None
    assert _get_tflite_attr(op_mock, "ADD") == {}


def test_tflite_all_options_parsing():
    from unittest.mock import MagicMock, patch
    import numpy as np
    import tflite
    from neuron_toolkit.backends.tflite.utils import _get_tflite_attr

    # Mock the options table returned by BuiltinOptions()
    options_mock = MagicMock()
    options_mock.Bytes = b""
    options_mock.Pos = 0

    op_mock = MagicMock()
    op_mock.BuiltinOptions.return_value = options_mock
    op_mock.BuiltinOptions2Type.return_value = 0

    # 1. Test Conv2DOptions (nn category)
    op_mock.BuiltinOptionsType.return_value = tflite.BuiltinOptions.Conv2DOptions
    with patch("tflite.Conv2DOptions.Conv2DOptions") as mock_class:
        mock_opt = mock_class.return_value
        mock_opt.StrideW.return_value = 2
        mock_opt.StrideH.return_value = 3
        mock_opt.Padding.return_value = 1  # Padding.VALID
        mock_opt.FusedActivationFunction.return_value = 1  # RELU
        mock_opt.DilationHFactor.return_value = 4
        mock_opt.DilationWFactor.return_value = 5
        mock_opt.QuantizedBiasType.return_value = 0

        attrs = _get_tflite_attr(op_mock, "CONV_2D")
        assert attrs["stride_w"] == 2
        assert attrs["stride_h"] == 3
        assert attrs["padding"] == "VALID"
        assert attrs["fused_activation_function"] == "RELU"
        assert attrs["dilation_h_factor"] == 4
        assert attrs["dilation_w_factor"] == 5

    # 2. Test SoftmaxOptions (activation category)
    op_mock.BuiltinOptionsType.return_value = tflite.BuiltinOptions.SoftmaxOptions
    with patch("tflite.SoftmaxOptions.SoftmaxOptions") as mock_class:
        mock_opt = mock_class.return_value
        mock_opt.Beta.return_value = 1.5

        attrs = _get_tflite_attr(op_mock, "SOFTMAX")
        assert attrs["beta"] == 1.5

    # 3. Test ReshapeOptions (tensor category with vector attribute)
    op_mock.BuiltinOptionsType.return_value = tflite.BuiltinOptions.ReshapeOptions
    with patch("tflite.ReshapeOptions.ReshapeOptions") as mock_class:
        mock_opt = mock_class.return_value
        mock_opt.NewShapeAsNumpy.return_value = np.array([1, 2, 3])

        attrs = _get_tflite_attr(op_mock, "RESHAPE")
        assert attrs["new_shape"] == [1, 2, 3]

    # 4. Test AddOptions (arithmetic category)
    op_mock.BuiltinOptionsType.return_value = tflite.BuiltinOptions.AddOptions
    with patch("tflite.AddOptions.AddOptions") as mock_class:
        mock_opt = mock_class.return_value
        mock_opt.FusedActivationFunction.return_value = 1  # RELU
        mock_opt.PotScaleInt16.return_value = 0

        attrs = _get_tflite_attr(op_mock, "ADD")
        assert attrs["fused_activation_function"] == "RELU"
        assert attrs["pot_scale_int16"] == 0

    # 5. Test empty options class (should immediately return {})
    op_mock.BuiltinOptionsType.return_value = tflite.BuiltinOptions.ZerosLikeOptions
    attrs = _get_tflite_attr(op_mock, "ZEROS_LIKE")
    assert attrs == {}



