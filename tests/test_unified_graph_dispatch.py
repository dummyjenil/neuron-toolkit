"""Unified ``NeuronGraph`` entry point tests.

Covers the auto-detection of ONNX and TFLite backends from paths, bytes,
protobufs and already-parsed backends, plus ``load()``/``save()``/
``get_weights()`` and the fluent query shortcuts.
"""

import os
import shutil

import numpy as np
import onnx
import pytest

from neuron_toolkit import NeuronGraph, Pattern
from neuron_toolkit.backends.onnx.parser import ONNXParser
from neuron_toolkit.backends.tflite.parser import TFLiteParser
from neuron_toolkit.rewriter import NeuronRewriter

# ---------------------------------------------------------------------------
# Construction dispatch
# ---------------------------------------------------------------------------


def test_graph_from_onnx_path(simple_model):
    g = NeuronGraph(simple_model)
    assert isinstance(g, NeuronGraph)
    assert len(g.nodes) == 3
    assert g.query().op("Add").count() == 1


def test_graph_from_onnx_model_proto(simple_model):
    model = onnx.load(simple_model)
    g = NeuronGraph(model)
    assert len(g.nodes) == 3
    assert g.nodes[0].op_type == "Identity"


def test_graph_from_onnx_bytes(simple_model):
    data = onnx.load(simple_model).SerializeToString()
    g = NeuronGraph(data)
    assert len(g.nodes) == 3
    assert g.query().op("Mul").count() == 1


def test_graph_from_onnx_parser(simple_model):
    g = NeuronGraph(ONNXParser(simple_model))
    assert len(g.nodes) == 3
    assert g.query().op("Add").count() == 1


def test_graph_from_tflite_bytes(tflite_abs_bytes):
    g = NeuronGraph(tflite_abs_bytes)
    assert [n.op_type for n in g.nodes] == ["ABS"]
    assert g.shape_info["input"] == (1, "float32")


def test_graph_from_tflite_object(tflite_abs_bytes):
    import tflite

    model_obj = tflite.Model.GetRootAsModel(tflite_abs_bytes, 0)
    g = NeuronGraph(model_obj)
    assert [n.op_type for n in g.nodes] == ["ABS"]


def test_graph_from_tflite_parser(tflite_abs_path):
    g = NeuronGraph(TFLiteParser(tflite_abs_path))
    assert [n.op_type for n in g.nodes] == ["ABS"]


def test_graph_from_str_dispatches_by_extension(simple_model, tflite_abs_path):
    onnx_g = NeuronGraph(simple_model)
    assert onnx_g.nodes[0].op_type == "Identity"
    tflite_g = NeuronGraph(tflite_abs_path)
    assert tflite_g.nodes[0].op_type == "ABS"


def test_graph_bytes_without_tfl3_magic_is_treated_as_onnx(tflite_abs_bytes):
    data = bytearray(tflite_abs_bytes)
    data[4:8] = b"XXXX"
    with pytest.raises(Exception):
        NeuronGraph(bytes(data))


def test_load_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported model format"):
        NeuronGraph.load("/tmp/model.ir")


def test_load_supported_extensions(simple_model, tflite_abs_path, tmp_path):
    g = NeuronGraph.load(simple_model)
    assert len(g.nodes) == 3

    target = tmp_path / "model.bin"
    shutil.copy(tflite_abs_path, target)
    g = NeuronGraph.load(str(target))
    assert [n.op_type for n in g.nodes] == ["ABS"]


# ---------------------------------------------------------------------------
# Properties / tensor data
# ---------------------------------------------------------------------------


def test_graph_properties(simple_model):
    g = NeuronGraph(simple_model)
    assert set(g.tensor_map) == {"B", "C"}
    assert g.shape_info["output"] == (2, "float32")
    assert g.nodes


def test_graph_get_weights(simple_model):
    weights = NeuronGraph(simple_model).get_weights()
    assert set(weights) == {"B", "C"}
    assert isinstance(weights["B"], np.ndarray)


# ---------------------------------------------------------------------------
# Fluent shortcuts
# ---------------------------------------------------------------------------


def test_query_match_findall(simple_model):
    g = NeuronGraph(simple_model)
    assert g.query().op("Identity").count() == 1

    m = g.match(Pattern.op("Mul", Pattern.op("Add")))
    assert m is not None
    assert m.start.op_type == "Mul"

    assert len(g.findall(Pattern.op("Add"))) == 1


def test_graph_replace_shortcut(simple_model):
    g = NeuronGraph(simple_model)
    rw = g.replace(Pattern.op("Identity"), "Relu")
    assert isinstance(rw, NeuronRewriter)
    new_graph = NeuronGraph(rw.build())
    assert new_graph.query().op("Relu").count() == 1
    assert new_graph.query().op("Identity").count() == 0


def test_rewriter_shortcut_requires_edits(simple_model):
    g = NeuronGraph(simple_model)
    rw = g.rewriter()
    assert isinstance(rw, NeuronRewriter)
    with pytest.raises(ValueError, match="No edits staged"):
        rw.build()


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


def test_save_onnx(simple_model, tmp_path):
    g = NeuronGraph(simple_model)
    out = str(tmp_path / "saved.onnx")
    g.save(out)
    assert os.path.exists(out)
    assert len(NeuronGraph(out).nodes) == 3


def test_save_tflite_bytes_source(tflite_abs_bytes, tmp_path):
    g = NeuronGraph(tflite_abs_bytes)
    out = str(tmp_path / "saved.tflite")
    g.save(out)
    assert os.path.exists(out)
    assert NeuronGraph(out).nodes[0].op_type == "ABS"


def test_save_tflite_str_source(tflite_abs_path, tmp_path):
    g = NeuronGraph(tflite_abs_path)
    out = str(tmp_path / "copied.tflite")
    g.save(out)
    assert os.path.exists(out)


def test_save_tflite_object_source_raises(tflite_abs_bytes, tmp_path):
    import tflite

    model_obj = tflite.Model.GetRootAsModel(tflite_abs_bytes, 0)
    g = NeuronGraph(model_obj)
    with pytest.raises(ValueError, match="parsed object"):
        g.save(str(tmp_path / "x.tflite"))


# ---------------------------------------------------------------------------
# slice wrappers
# ---------------------------------------------------------------------------


def test_slice_and_slice_node(simple_model):
    g = NeuronGraph(simple_model)
    add = g.query().op("Add").single_node
    sliced = g.slice(add, add)
    assert len(sliced.nodes) == 1
    assert sliced.nodes[0].op_type == "Add"

    node_piece = g.slice_node("node_id")
    assert len(node_piece.nodes) == 1
    assert node_piece.nodes[0].op_type == "Identity"


def test_slice_invalid_endpoints_raise(simple_model):
    g = NeuronGraph(simple_model)
    with pytest.raises(ValueError, match="start or end"):
        g.slice("not_a_tensor", "also_not_a_tensor")
