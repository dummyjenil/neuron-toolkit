from onnx_toolkit.parser import ONNXParser


def test_parser_loading(simple_model):
    parser = ONNXParser(simple_model)
    assert len(parser.nodes) == 3
    assert "A" in parser.graph_inputs
    assert "output" in parser.graph_outputs
    assert "B" in parser.tensor_map
    assert "C" in parser.tensor_map

def test_parser_summary(simple_model):
    parser = ONNXParser(simple_model)
    summary = parser.summary()
    assert "Add" in summary
    assert "Mul" in summary
    assert "Nodes        : 3" in summary

def test_parser_find(simple_model):
    parser = ONNXParser(simple_model)
    query = parser.find()
    assert query.count() == 3
    assert set(query.op_types()) == {"Identity", "Add", "Mul"}
