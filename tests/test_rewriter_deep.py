import pytest

from onnx_toolkit.parser import ONNXParser
from onnx_toolkit.pattern import Pattern


def test_rewriter_replace(simple_model, tmp_path):
    parser = ONNXParser(simple_model)
    rewriter = parser.rewriter()

    # Let's replace Add and Mul with a single MyFusion op
    # (Identity(A) + B) * C -> MyFusion(Identity(A), B, C)

    node_add = parser.find().op("Add").first()
    node_mul = parser.find().op("Mul").first()
    assert node_add is not None
    assert node_mul is not None

    rewriter.replace(
        nodes=[node_add, node_mul],
        new_op="MyFusion",
        inputs=["A_id", "B", "C"],
        outputs=["output"],
        name="fused_node",
    )

    out_path = str(tmp_path / "rewritten.onnx")
    new_model = rewriter.build(out_path)

    assert len(new_model.graph.node) == 2  # Identity and MyFusion
    assert new_model.graph.node[1].op_type == "MyFusion"
    assert new_model.graph.node[1].name == "fused_node"


def test_rewriter_replace_from_result(simple_model, tmp_path):
    parser = ONNXParser(simple_model)

    # Pattern: Mul(Add)
    p = Pattern.op("Mul", Pattern.op("Add"))
    match = parser.pattern_detect(p, start_node="node_mul")
    assert match is not None

    rewriter = parser.rewriter()
    # Replace all nodes in match with "FusedAddMul"
    # End node is Add, start node is Mul.
    # replace_from_result will use Add's input (A_id) and Mul's output (output) by default if not specified?
    # Wait, let's check rewriter.py:87
    # ins = inputs or [result.end.output[0]] -> End node is Add, output[0] is add_out.
    # outs = outputs or [result.start.output[0]] -> Start node is Mul, output[0] is output.
    # So it will be FusedAddMul(add_out) -> output.
    # This is probably not what we want if we wanted to replace the WHOLE subgraph.
    # We should probably specify inputs.

    rewriter.replace_from_result(match, "FusedAddMul", inputs=["A_id", "B", "C"])

    new_model = rewriter.build()
    assert len(new_model.graph.node) == 2  # Identity and FusedAddMul


def test_rewriter_delete(simple_model):
    parser = ONNXParser(simple_model)
    rewriter = parser.rewriter()

    node_mul = parser.find().op("Mul").first()
    assert node_mul is not None
    rewriter.delete([node_mul])

    new_model = rewriter.build()
    assert len(new_model.graph.node) == 2
    assert "Mul" not in [n.op_type for n in new_model.graph.node]


def test_rewriter_insert_before(simple_model):
    parser = ONNXParser(simple_model)
    rewriter = parser.rewriter()

    node_mul = parser.find().op("Mul").first()
    assert node_mul is not None
    # Insert a Relu before Mul
    # Mul currently takes add_out and C.
    # Let's insert Relu(add_out) -> relu_out, and change Mul to take relu_out.
    # Wait, insert_before doesn't automatically rewire the target_node.
    # The user must do that manually?
    # rewriter.py:113 "The caller is responsible for ensuring *outputs* matches the inputs that *target_node* expects."
    # So I should probably replace the target_node too if I change its inputs.

    rewriter.insert_before(
        node_mul, "Relu", inputs=["add_out"], outputs=["relu_out"], name="new_relu"
    )
    # Now I must update node_mul to take relu_out instead of add_out
    rewriter.replace(
        nodes=[node_mul],
        new_op="Mul",
        inputs=["relu_out", "C"],
        outputs=["output"],
        name="node_mul",  # keep same name
    )

    new_model = rewriter.build()
    assert len(new_model.graph.node) == 4  # Identity, Add, Relu, Mul
    assert new_model.graph.node[2].op_type == "Relu"
    assert new_model.graph.node[3].input[0] == "relu_out"


def test_rewriter_reset(simple_model):
    parser = ONNXParser(simple_model)
    rewriter = parser.rewriter()
    rewriter.delete(parser.nodes)
    rewriter.reset()
    with pytest.raises(ValueError, match="No edits staged"):
        rewriter.build()


def test_rewriter_no_edits(simple_model):
    parser = ONNXParser(simple_model)
    rewriter = parser.rewriter()
    with pytest.raises(ValueError, match="No edits staged"):
        rewriter.build()
