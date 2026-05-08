import numpy as np
import onnx
from onnx import TensorProto, helper

from onnx_toolkit._utils import (
    _ONNX_DTYPE_TO_NP,
    _attr_value,
    _build_shape_info,
    _node_attrs,
)


def test_attr_value_all_types():
    # FLOAT
    attr_f = helper.make_attribute("f", 1.5)
    assert _attr_value(attr_f) == 1.5

    # INT
    attr_i = helper.make_attribute("i", 42)
    assert _attr_value(attr_i) == 42

    # STRING
    attr_s = helper.make_attribute("s", b"hello")
    assert _attr_value(attr_s) == "hello"

    # TENSOR
    tensor_val = np.array([1, 2, 3], dtype=np.float32)
    attr_t = helper.make_attribute(
        "t", helper.make_tensor("t", TensorProto.FLOAT, [3], tensor_val.tobytes(), raw=True)
    )
    np.testing.assert_array_equal(_attr_value(attr_t), tensor_val)

    # FLOATS
    attr_fs = helper.make_attribute("fs", [1.0, 2.0])
    assert _attr_value(attr_fs) == [1.0, 2.0]

    # INTS
    attr_is = helper.make_attribute("is", [1, 2])
    assert _attr_value(attr_is) == [1, 2]

    # STRINGS
    attr_ss = helper.make_attribute("ss", [b"a", b"b"])
    assert _attr_value(attr_ss) == ["a", "b"]


def test_attr_value_unrecognized(caplog):
    # Manually create an AttributeProto with an unsupported type if possible
    attr = onnx.AttributeProto()
    attr.name = "unknown"
    attr.type = onnx.AttributeProto.SPARSE_TENSOR  # Likely not handled in scalar_extractors

    with caplog.at_level("DEBUG", logger="onnx_toolkit"):
        val = _attr_value(attr)
        assert val is None
        assert "unrecognised attribute type" in caplog.text


def test_node_attrs():
    node = helper.make_node(
        "Conv",
        ["X", "W"],
        ["Y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        group=1,
        dilations=[1, 1],
    )
    attrs = _node_attrs(node)
    assert attrs == {
        "kernel_shape": [3, 3],
        "strides": [1, 1],
        "pads": [1, 1, 1, 1],
        "group": 1,
        "dilations": [1, 1],
    }


def test_build_shape_info():
    # Create a model with various types and shapes
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [1, 1000])

    node = helper.make_node("Identity", ["X"], ["Y_id"])
    Y_id = helper.make_tensor_value_info("Y_id", TensorProto.INT64, [1, 1000])

    graph = helper.make_graph([node], "test", [X], [Y], value_info=[Y_id])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    # build_shape_info expects inferred model (it uses value_info and output)
    info = _build_shape_info(model)

    assert "Y_id" in info
    assert info["Y_id"] == (2, "int64")

    assert "Y" in info
    assert info["Y"] == (2, "int64")


def test_build_shape_info_no_shape():
    # Value info without shape
    vi = onnx.ValueInfoProto()
    vi.name = "Z"
    vi.type.tensor_type.elem_type = TensorProto.FLOAT

    graph = helper.make_graph([], "test", [], [vi])
    model = helper.make_model(graph)

    info = _build_shape_info(model)
    assert info["Z"] == (None, "float32")


def test_onnx_dtype_mapping():
    assert _ONNX_DTYPE_TO_NP[TensorProto.FLOAT] == "float32"
    assert _ONNX_DTYPE_TO_NP[TensorProto.INT64] == "int64"
    assert _ONNX_DTYPE_TO_NP[TensorProto.BOOL] == "bool"
