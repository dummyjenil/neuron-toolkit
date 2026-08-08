"""Tests for ONNX-vs-PyTorch output comparison (compare_outputs / verify_against_pytorch)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from onnx import TensorProto, helper  # noqa: E402

from neuron_toolkit import NeuronGraph  # noqa: E402


def _two_input_model(path) -> None:
    """x*y + x + y with graph inputs A, B and output Z."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 4])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 4])
    mul = helper.make_node("Mul", ["A", "B"], ["m"], name="mul0")
    add = helper.make_node("Add", ["m", "A"], ["i"], name="add0")
    add2 = helper.make_node("Add", ["i", "B"], ["Z"], name="add1")
    graph = helper.make_graph([mul, add, add2], "two_inputs", [A, B], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path.write_bytes(model.SerializeToString())


def _two_output_model(path) -> None:
    """Outputs S = A + B and P = A * B."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 4])
    S = helper.make_tensor_value_info("S", TensorProto.FLOAT, [1, 4])
    P = helper.make_tensor_value_info("P", TensorProto.FLOAT, [1, 4])
    add = helper.make_node("Add", ["A", "B"], ["S"], name="add0")
    mul = helper.make_node("Mul", ["A", "B"], ["P"], name="mul0")
    graph = helper.make_graph([add, mul], "two_outputs", [A, B], [S, P])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path.write_bytes(model.SerializeToString())


def _ref_two(x, y):
    return x * y + x + y


def _ref_wrong(x, y):
    return x * y - x - y


def test_verify_against_pytorch_matches(tmp_path):
    p = tmp_path / "two.onnx"
    _two_input_model(p)
    graph = NeuronGraph(str(p))

    a = torch.randn(1, 4)
    b = torch.randn(1, 4)

    @graph.verify_against_pytorch(["A"], ["Z"])
    def ref(x, y):
        return x * y + x + y

    py_out = ref(a, b)

    report = ref.similarity_report
    assert "Z" in report
    assert report["Z"]["all_close"] is True
    assert report["Z"]["shape_match"] is True
    assert torch.equal(py_out, a * b + a + b)


def test_verify_against_pytorch_detects_mismatch(tmp_path):
    p = tmp_path / "two.onnx"
    _two_input_model(p)
    graph = NeuronGraph(str(p))

    a = torch.randn(1, 4)
    b = torch.randn(1, 4)

    @graph.verify_against_pytorch(["A"], ["Z"])
    def ref(x, y):
        return x * y - x - y

    ref(a, b)
    assert ref.similarity_report["Z"]["all_close"] is False
    assert ref.similarity_report["Z"]["shape_match"] is True


def test_verify_against_pytorch_kwargs_input(tmp_path):
    p = tmp_path / "two.onnx"
    _two_input_model(p)
    graph = NeuronGraph(str(p))

    a = torch.randn(1, 4)
    b = torch.randn(1, 4)

    @graph.verify_against_pytorch(["A"], ["Z"])
    def ref(x, y):
        return x * y + x + y

    ref(a, y=torch.as_tensor(b))
    assert ref.similarity_report["Z"]["all_close"] is True


def test_compare_outputs_exact_names(tmp_path):
    p = tmp_path / "two.onnx"
    _two_input_model(p)
    graph = NeuronGraph(str(p))

    a = np.random.randn(1, 4).astype(np.float32)
    b = np.random.randn(1, 4).astype(np.float32)

    def ref(x, y):
        return x * y + x + y

    report = graph.compare_outputs(["A"], ["Z"], ref, {"A": a, "B": b})
    assert report["Z"]["all_close"] is True


def test_compare_outputs_multi_output_tuple(tmp_path):
    p = tmp_path / "two_out.onnx"
    _two_output_model(p)
    graph = NeuronGraph(str(p))

    a = np.random.randn(1, 4).astype(np.float32)
    b = np.random.randn(1, 4).astype(np.float32)

    def ref(x, y):
        return x + y, x * y

    report = graph.compare_outputs(["A"], ["S", "P"], ref, {"A": a, "B": b})
    assert report["S"]["all_close"] is True
    assert report["P"]["all_close"] is True


def test_compare_outputs_object_endpoints(tmp_path):
    p = tmp_path / "two.onnx"
    _two_input_model(p)
    graph = NeuronGraph(str(p))

    a = np.random.randn(1, 4).astype(np.float32)
    b = np.random.randn(1, 4).astype(np.float32)
    end_node = graph.nodes[-1]  # Add producing Z

    def ref(x, y):
        return x * y + x + y

    report = graph.compare_outputs(["A"], [end_node], ref, {"A": a, "B": b})
    assert report["Z"]["all_close"] is True
