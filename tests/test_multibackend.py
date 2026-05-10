from neuron_toolkit import NeuronGraph, Pattern
from neuron_toolkit.backends.tflite.parser import TFLiteNode


def test_onnx_backend_loading(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    # Create a simple ONNX model
    node = helper.make_node("Relu", ["X"], ["Y"], name="test_relu")
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])
    graph = helper.make_graph([node], "test", [X], [Y])
    model = helper.make_model(graph)

    onnx_path = tmp_path / "model.onnx"
    onnx.save(model, str(onnx_path))

    # Load using NeuronGraph
    g = NeuronGraph.load(str(onnx_path))
    assert g is not None
    assert len(g.nodes) == 1
    assert g.nodes[0].op_type == "Relu"

    # Test Query
    q = g.query().op("Relu")
    assert q.count() == 1
    assert q.first().name == "test_relu"


def test_tflite_backend_mock(monkeypatch):
    # Since our TFLite parser is a real implementation now, we mock _load_model
    from neuron_toolkit.backends.tflite.parser import TFLiteParser

    monkeypatch.setattr(TFLiteParser, "_load_model", lambda self: None)

    # Create a dummy parser
    parser = TFLiteParser("dummy.tflite")

    # Manually add a mock node
    mock_node = TFLiteNode(
        op_type="CONV_2D",
        name="conv1",
        input=["input", "weight"],
        output=["output"],
        attrs={"stride": 1},
    )
    parser.nodes = [mock_node]
    parser.graph_inputs = {"input"}
    parser.graph_outputs = {"output"}

    # Create NeuronGraph with this backend
    g = NeuronGraph(parser)

    # Test Query
    q = g.query().op("CONV_2D")
    assert q.count() == 1
    assert q.first().name == "conv1"

    # Test Attribute access (generic)
    assert q.attr("stride", 1).count() == 1
    assert q.attr("stride", 2).count() == 0


def test_backend_agnostic_pattern_matching(monkeypatch):
    from neuron_toolkit.backends.tflite.parser import TFLiteParser

    monkeypatch.setattr(TFLiteParser, "_load_model", lambda self: None)

    # TFLite mock nodes
    n1 = TFLiteNode("Relu", "r1", ["x"], ["y"])
    n2 = TFLiteNode("Add", "a1", ["y", "z"], ["out"])

    parser = TFLiteParser("dummy.tflite")
    parser.nodes = [n1, n2]

    g = NeuronGraph(parser)

    # Pattern: Relu -> Add
    relu_pat = Pattern.op("Relu").capture("child")
    add_pat = Pattern.op("Add", relu_pat)

    matches = g.findall(add_pat)
    assert len(matches) == 1
    assert matches[0].start.op_type == "Add"
    assert matches[0].bindings["child"].op_type == "Relu"

    # Using captured names
    p = Pattern.op("Relu").capture("r")
    p2 = Pattern.op("Add", p).capture("a")

    m = g.match(p2)
    assert m is not None
    assert m.bindings["r"].name == "r1"
    assert m.bindings["a"].name == "a1"


def test_neuron_names_export():
    import neuron_toolkit

    assert hasattr(neuron_toolkit, "NeuronGraph")
    assert hasattr(neuron_toolkit, "NeuronQuery")
    assert hasattr(neuron_toolkit, "NeuronRewriter")  # Alias for NeuronRewriter
