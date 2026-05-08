
import pytest
import onnx
from onnx_toolkit.parser import ONNXParser
from onnx_toolkit.query import ONNXQuery
from onnx_toolkit.pattern import Pattern

def test_parser_init(simple_model):
    parser = ONNXParser(simple_model)
    assert len(parser.nodes) == 3
    assert "A" in parser.graph_inputs
    assert "output" in parser.graph_outputs
    assert "B" in parser.tensor_map
    assert "C" in parser.tensor_map
    assert "A_id" in parser.shape_info

def test_parser_no_infer(simple_model):
    # If we disable shape inference, shape_info might still have some data from graph.output
    # but maybe less than if we infer.
    parser = ONNXParser(simple_model, infer_shapes=False)
    # The simple_model in conftest already has output shape info
    assert "output" in parser.shape_info

def test_parser_find(simple_model):
    parser = ONNXParser(simple_model)
    query = parser.find()
    assert isinstance(query, ONNXQuery)
    assert len(query.all_nodes) == 3

def test_parser_pattern_detect(simple_model):
    parser = ONNXParser(simple_model)
    # Pattern: Mul(Add, const)
    # We should start matching from the end of the chain
    p = Pattern.op("Mul", Pattern.op("Add"), Pattern.const(2.0))
    match = parser.pattern_detect(p, start_node="node_mul")
    assert match is not None
    assert match.start.op_type == "Mul"
    assert "node_add" in [n.name for n in match.nodes]

def test_parser_summary(simple_model):
    parser = ONNXParser(simple_model)
    summary = parser.summary()
    assert "Nodes        : 3" in summary
    assert "Identity" in summary
    assert "Add" in summary
    assert "Mul" in summary

def test_parser_invalid_path():
    with pytest.raises(Exception): # onnx.load raises FileNotFoundError or similar
        ONNXParser("non_existent.onnx")

def test_parser_corrupt_model(tmp_path):
    corrupt_path = tmp_path / "corrupt.onnx"
    corrupt_path.write_text("not an onnx model")
    with pytest.raises(Exception):
        ONNXParser(str(corrupt_path))
