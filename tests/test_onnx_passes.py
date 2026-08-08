"""Tests for the ONNX GraphPasses (Conv+BN fusion) and ONNXRewriter helpers."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from neuron_toolkit import NeuronGraph

pytest.importorskip("onnxruntime")


def _conv_bn_model(
    path,
    *,
    conv_bias: bool = True,
    spatial: int = 5,
    training_mode: int | None = None,
    group: int = 1,
):
    """Conv 3x3 -> BatchNormalization with configurable params."""
    rng = np.random.default_rng(7)
    oc = 2
    ic = 2 if group > 1 else 1
    x_shape = [1, ic, spatial, spatial]
    out = spatial - 2

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, oc, out, out])

    W = rng.normal(size=(oc, ic // group, 3, 3)).astype(np.float32)
    scale = rng.uniform(0.5, 2.0, (oc,)).astype(np.float32)
    beta = rng.normal(size=(oc,)).astype(np.float32)
    mean = rng.normal(size=(oc,)).astype(np.float32)
    var = rng.uniform(0.5, 1.5, (oc,)).astype(np.float32)

    inits = [
        numpy_helper.from_array(W, "W"),
        numpy_helper.from_array(scale, "scale"),
        numpy_helper.from_array(beta, "beta"),
        numpy_helper.from_array(mean, "mean"),
        numpy_helper.from_array(var, "var"),
    ]
    conv_inputs = ["X", "W"]
    if conv_bias:
        cb = rng.normal(size=(oc,)).astype(np.float32)
        inits.append(numpy_helper.from_array(cb, "cb"))
        conv_inputs = ["X", "W", "cb"]

    conv_attrs = {"kernel_shape": [3, 3]}
    if group > 1:
        conv_attrs["group"] = group
    conv = helper.make_node(
        "Conv", conv_inputs, ["conv_out"], name="conv0", **conv_attrs
    )
    bn_attrs = {}
    if training_mode is not None:
        bn_attrs["training_mode"] = training_mode
    bn = helper.make_node(
        "BatchNormalization",
        ["conv_out", "scale", "beta", "mean", "var"],
        ["Y"],
        name="bn0",
        epsilon=1e-5,
        **bn_attrs,
    )
    graph = helper.make_graph([conv, bn], "conv_bn", [X], [Y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path.write_bytes(model.SerializeToString())
    return x_shape


def _run(model_path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path))
    return session.run(None, {"X": x})[0]


def test_fuse_conv_bn_folds_to_single_conv(tmp_path):
    p = tmp_path / "model.onnx"
    x_shape = _conv_bn_model(p)
    graph = NeuronGraph(str(p))
    assert [n.op_type for n in graph.nodes] == ["Conv", "BatchNormalization"]

    fused = graph.passes.fuse_conv_bn().build()
    onnx.checker.check_model(fused)
    assert [n.op_type for n in fused.graph.node] == ["Conv"]
    assert "bn0_weight" in {i.name for i in fused.graph.initializer}
    assert "bn0_bias" in {i.name for i in fused.graph.initializer}

    fused_path = tmp_path / "fused.onnx"
    fused_path.write_bytes(fused.SerializeToString())
    x = np.random.default_rng(1).normal(size=x_shape).astype(np.float32)
    np.testing.assert_allclose(_run(p, x), _run(fused_path, x), atol=1e-6)


def test_fuse_conv_bn_without_conv_bias(tmp_path):
    p = tmp_path / "model.onnx"
    x_shape = _conv_bn_model(p, conv_bias=False)
    graph = NeuronGraph(str(p))

    fused = graph.passes.fuse_conv_bn().build()
    onnx.checker.check_model(fused)
    assert [n.op_type for n in fused.graph.node] == ["Conv"]

    x = np.random.default_rng(2).normal(size=x_shape).astype(np.float32)
    (tmp_path / "fused.onnx").write_bytes(fused.SerializeToString())
    np.testing.assert_allclose(_run(p, x), _run(tmp_path / "fused.onnx", x), atol=1e-6)


def test_fuse_conv_bn_grouped_conv(tmp_path):
    p = tmp_path / "model.onnx"
    x_shape = _conv_bn_model(p, group=2)
    graph = NeuronGraph(str(p))

    fused = graph.passes.fuse_conv_bn().build()
    onnx.checker.check_model(fused)
    assert [n.op_type for n in fused.graph.node] == ["Conv"]
    conv_node = fused.graph.node[0]
    assert any(a.name == "group" and a.i == 2 for a in conv_node.attribute)

    x = np.random.default_rng(3).normal(size=x_shape).astype(np.float32)
    (tmp_path / "fused.onnx").write_bytes(fused.SerializeToString())
    np.testing.assert_allclose(_run(p, x), _run(tmp_path / "fused.onnx", x), atol=1e-6)


def test_fuse_conv_bn_training_mode_skips(tmp_path):
    p = tmp_path / "model.onnx"
    _conv_bn_model(p, training_mode=1)
    graph = NeuronGraph(str(p))

    rewriter = graph.passes.fuse_conv_bn()
    with pytest.raises(ValueError, match="No edits staged"):
        rewriter.build()
    assert [n.op_type for n in graph.nodes] == ["Conv", "BatchNormalization"]


def test_fuse_conv_bn_no_match_raises(tmp_path):
    p = tmp_path / "ident.onnx"
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["A"], ["Z"])
    graph = helper.make_graph([node], "identity", [A], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    p.write_bytes(model.SerializeToString())

    graph = NeuronGraph(str(p))
    rewriter = graph.passes.fuse_conv_bn()
    with pytest.raises(ValueError, match="No edits staged"):
        rewriter.build()


def test_register_initializer_adds_constant(tmp_path):
    from onnx import helper as h

    A = h.make_tensor_value_info("A", TensorProto.FLOAT, [1])
    Z = h.make_tensor_value_info("Z", TensorProto.FLOAT, [1])
    node = h.make_node("Identity", ["A"], ["Z"])
    graph = h.make_graph([node], "identity", [A], [Z])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    p = tmp_path / "ident.onnx"
    p.write_bytes(model.SerializeToString())

    graph = NeuronGraph(str(p))
    rewriter = graph.rewriter()
    rewriter.register_initializer("extra", np.array([3.0], dtype=np.float32))
    rewriter.replace(graph.nodes, "Add", inputs=["A", "extra"], outputs=["Z"])
    new = rewriter.build()

    names = {i.name for i in new.graph.initializer}
    assert "extra" in names
    assert new.graph.node[0].op_type == "Add"
    assert list(new.graph.node[0].input) == ["A", "extra"]
    onnx.checker.check_model(new)
