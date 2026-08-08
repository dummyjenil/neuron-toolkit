"""Tests for the ONNX ``LazyTensorMap`` initializer map.

Verifies the lazy-loading behavior users rely on when inspecting model
parameters through ``NeuronGraph.tensor_map`` / ``parser.tensor_map``.
"""

import numpy as np
import pytest

from neuron_toolkit.backends.onnx.parser import LazyTensorMap, ONNXParser


def test_tensor_map_lazy_and_cached(simple_model):
    parser = ONNXParser(simple_model)
    tm = parser.tensor_map

    assert isinstance(tm, LazyTensorMap)
    # Not loaded until requested.
    assert tm._cache == {}
    assert "B" in tm
    # First access parses the initializer.
    b = tm["B"]
    assert isinstance(b, np.ndarray)
    assert b.shape == (1, 3)
    np.testing.assert_array_equal(b, [[1, 2, 3]])
    # Second access hits the cache (same object, no re-parse).
    assert tm["B"] is b
    assert "B" in tm._cache


def test_tensor_map_mapping_protocol(simple_model):
    tm = ONNXParser(simple_model).tensor_map

    assert len(tm) == 2
    assert set(tm.keys()) == {"B", "C"}
    assert set(iter(tm)) == {"B", "C"}
    items = dict(tm.items())
    assert set(items) == {"B", "C"}
    assert isinstance(items["C"], np.ndarray)
    # Membership works for names that were not materialized yet.
    assert "C" in tm
    assert "missing" not in tm


def test_tensor_map_get_and_missing(simple_model):
    tm = ONNXParser(simple_model).tensor_map
    np.testing.assert_array_equal(tm.get("B"), [[1, 2, 3]])
    assert tm.get("does_not_exist") is None
    assert tm.get("does_not_exist", "fallback") == "fallback"

    with pytest.raises(KeyError):
        tm["does_not_exist"]


def test_tensor_map_int_and_bool_types(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1])
    fixed = helper.make_tensor(
        "t0", TensorProto.INT64, [1], np.array([0], dtype=np.int64).tobytes(), raw=True
    )
    node = helper.make_node("Less", ["A", "t0"], ["Y"])
    graph = helper.make_graph(
        [node],
        "g",
        [A],
        [helper.make_tensor_value_info("Y", TensorProto.BOOL, [1])],
        [fixed],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = str(tmp_path / "types.onnx")
    onnx.save(model, path)

    tm = ONNXParser(path).tensor_map
    assert tm["t0"].dtype == np.int64
    np.testing.assert_array_equal(tm["t0"], [0])
