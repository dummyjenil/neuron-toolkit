from onnx_toolkit.parser import ONNXParser
from onnx_toolkit.pattern import Pattern


def test_rewriter_replace(simple_model, tmp_path):
    parser = ONNXParser(simple_model)

    # Replace Add with Sub
    add_node = parser.find().find_by_op_type("Add").single_node

    rw = parser.rewriter()
    rw.replace([add_node], "Sub", inputs=list(add_node.input), outputs=list(add_node.output))

    new_model = rw.build()
    assert len(new_model.graph.node) == 3
    op_types = [n.op_type for n in new_model.graph.node]
    assert "Sub" in op_types
    assert "Add" not in op_types


def test_rewriter_replace_from_result(simple_model):
    parser = ONNXParser(simple_model)

    pat = Pattern.op("Add", Pattern.any(), Pattern.any())
    # Note: simple_model has Add(Identity, B).
    # B is an initializer, but Pattern.any() only matches nodes.
    # Wait! If I use Pattern.any() it will fail to match B.
    # I should use Pattern.const() or Pattern.any() (if I fix the library).
    # For now, let's use a pattern that matches.
    import numpy as np

    B_val = np.array([[1, 2, 3]], dtype=np.float32)
    pat = Pattern.op("Add", Pattern.any(), Pattern.const(B_val))

    res = parser.pattern_detect(pat, start_node="node_add")
    assert res is not None

    rw = parser.rewriter()
    rw.replace_from_result(res, "CustomOp")

    new_model = rw.build()
    op_types = [n.op_type for n in new_model.graph.node]
    assert "CustomOp" in op_types
    assert "Add" not in op_types


def test_rewriter_delete(complex_model):
    parser = ONNXParser(complex_model)

    # Delete Tanh node
    tanh_node = parser.find().find_by_op_type("Tanh").single_node

    rw = parser.rewriter()
    rw.delete([tanh_node])

    new_model = rw.build()
    assert len(new_model.graph.node) == 5
    assert "Tanh" not in [n.op_type for n in new_model.graph.node]


def test_rewriter_insert_before(simple_model):
    parser = ONNXParser(simple_model)

    add_node = parser.find().find_by_op_type("Add").single_node

    rw = parser.rewriter()
    rw.insert_before(add_node, "Identity", inputs=["A"], outputs=["A_ident"])

    new_model = rw.build()
    assert len(new_model.graph.node) == 4
    assert "Identity" in [n.op_type for n in new_model.graph.node]
