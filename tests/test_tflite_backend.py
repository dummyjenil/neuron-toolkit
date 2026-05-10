from unittest.mock import MagicMock, mock_open, patch

import pytest
import tflite

from neuron_toolkit.backends.tflite.parser import TFLiteParser
from neuron_toolkit.pattern import Pattern


class MockBuiltinOperator:
    ADD = 0
    CUSTOM = 255

    # Add other attributes that might be accessed
    def __iter__(self):
        return iter({"ADD": 0}.items())


@pytest.fixture
def mock_tflite_model():
    mock_model = MagicMock()
    mock_subgraph = MagicMock()
    mock_model.Subgraphs.return_value = mock_subgraph

    # Setup subgraph
    mock_subgraph.TensorsLength.return_value = 2

    t1 = MagicMock()
    t1.Name.return_value = b"input"
    t1.ShapeLength.return_value = 2
    t1.Shape.side_effect = lambda j: [1, 10][j]
    t1.Type.return_value = 0  # FLOAT32

    t2 = MagicMock()
    t2.Name.return_value = b"output"
    t2.ShapeLength.return_value = 2
    t2.Shape.side_effect = lambda j: [1, 5][j]
    t2.Type.return_value = 0

    def get_tensor(i):
        if i == 0:
            return t1
        return t2

    mock_subgraph.Tensors.side_effect = get_tensor

    # Setup operators
    mock_subgraph.OperatorsLength.return_value = 1
    mock_op = MagicMock()
    mock_subgraph.Operators.return_value = mock_op
    mock_op.OpcodeIndex.return_value = 0
    mock_op.InputsLength.return_value = 1
    mock_op.Inputs.return_value = 0
    mock_op.OutputsLength.return_value = 1
    mock_op.Outputs.return_value = 1
    mock_op.BuiltinOptions.return_value = None

    mock_opcode = MagicMock()
    mock_model.OperatorCodes.return_value = mock_opcode
    mock_opcode.BuiltinCode.return_value = 0  # ADD

    # We need to mock BuiltinOperator.__dict__ because it's used in a loop in parser.py
    # Since we can't patch __dict__ directly, we can mock the whole class
    mock_builtin_op = MagicMock()
    mock_builtin_op.ADD = 0
    mock_builtin_op.CUSTOM = 255
    # This is what parser.py does: [k for k, v in tflite.BuiltinOperator.__dict__.items() if v == builtin_code]
    # So we MUST mock __dict__ or the class.
    # Actually, we can patch 'tflite.BuiltinOperator' with an object that has a __dict__

    with patch.object(tflite.Model, "GetRootAsModel", return_value=mock_model):
        with patch("tflite.BuiltinOperator") as mocked_builtin_op:
            mocked_builtin_op.__dict__ = {"ADD": 0, "CUSTOM": 255}
            yield mock_model


def test_tflite_parser_load_mocked(mock_tflite_model):
    with patch("builtins.open", mock_open(read_data=b"dummy")):
        parser = TFLiteParser("dummy.tflite")
        assert len(parser.nodes) == 1
        assert parser.nodes[0].op_type == "ADD"
        assert "input" in parser.nodes[0].input
        assert "output" in parser.nodes[0].output


def test_tflite_query_mocked(mock_tflite_model):
    with patch("builtins.open", mock_open(read_data=b"dummy")):
        parser = TFLiteParser("dummy.tflite")
        q = parser.find().op("ADD")
        assert q.count() == 1


def test_tflite_rewriter_staging_mocked(mock_tflite_model):
    with patch("builtins.open", mock_open(read_data=b"dummy")):
        parser = TFLiteParser("dummy.tflite")
        rw = parser.rewriter()
        node = parser.nodes[0]
        rw.replace([node], "MUL", node.input, node.output)
        path = rw.build("out.tflite")
        assert path == "out.tflite"


def test_tflite_pattern_detect_mocked(mock_tflite_model):
    with patch("builtins.open", mock_open(read_data=b"dummy")):
        parser = TFLiteParser("dummy.tflite")
        # Define a pattern that matches our ADD node
        pat = Pattern.op("ADD")
        # To match, we must provide a start node or use find_all
        node = parser.nodes[0]
        match = parser.pattern_detect(pat, start_node=node)
        assert match is not None
        assert match.start.op_type == "ADD"
