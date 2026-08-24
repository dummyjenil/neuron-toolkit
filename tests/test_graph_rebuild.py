"""Tests for rebuilding graph from JSON + Safetensors and in-place weight replacement."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

import neuron_toolkit
from neuron_toolkit._utils import _GraphShim
from neuron_toolkit.exporter import (
    build_onnx_model_from_json,
    export_graph_dict,
    load_graph_from_json,
)
from neuron_toolkit.graph import NeuronGraph


class TestGraphRebuild(unittest.TestCase):
    """Test suite for reconstructing models from JSON and Safetensors."""

    def setUp(self):
        # Create a mock graph: input_0 -> Conv1 -> Relu1 -> Add1 -> output_0
        node1 = SimpleNamespace(
            op_type="Conv",
            name="conv1",
            input=["input_0", "conv1.weight", "conv1.bias"],
            output=["conv1_out"],
            attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1]},
        )
        node2 = SimpleNamespace(
            op_type="Relu",
            name="relu1",
            input=["conv1_out"],
            output=["relu1_out"],
            attrs={},
        )
        node3 = SimpleNamespace(
            op_type="Add",
            name="add1",
            input=["relu1_out", "add_const"],
            output=["output_0"],
            attrs={},
        )

        self.tensor_map = {
            "conv1.weight": np.ones((8, 3, 3, 3), dtype=np.float32) * 0.75,
            "conv1.bias": np.full((8,), 0.25, dtype=np.float32),
            "add_const": np.array([2.5], dtype=np.float32),
        }
        self.nodes = [node1, node2, node3]
        self.shape_info = {
            "input_0": (4, "float32"),
            "conv1_out": (4, "float32"),
            "relu1_out": (4, "float32"),
            "output_0": (4, "float32"),
            "conv1.weight": (4, "float32"),
            "conv1.bias": (1, "float32"),
            "add_const": (1, "float32"),
        }

        self.backend = _GraphShim(
            nodes=self.nodes,
            tensor_map=self.tensor_map,
            shape_info=self.shape_info,
        )
        self.backend.graph_inputs = {"input_0"}
        self.backend.graph_outputs = {"output_0"}
        self.graph = NeuronGraph(self.backend)

    def test_rebuild_from_json_with_random_initialization(self):
        """Test rebuilding graph from JSON only with random weights."""
        json_dict = self.graph.to_dict()

        # Rebuild graph from dict with seed
        rebuilt = NeuronGraph.from_json(json_dict, seed=42)

        assert len(rebuilt.nodes) == 3
        node_ops = [n.op_type for n in rebuilt.nodes]
        assert node_ops == ["Conv", "Relu", "Add"]

        # Check weights exist and have correct shapes/dtypes
        assert "conv1.weight" in rebuilt.tensor_map
        assert "conv1.bias" in rebuilt.tensor_map
        assert "add_const" in rebuilt.tensor_map

        w = rebuilt.tensor_map["conv1.weight"]
        b = rebuilt.tensor_map["conv1.bias"]
        c = rebuilt.tensor_map["add_const"]

        assert w.shape == (8, 3, 3, 3)
        assert w.dtype == np.float32
        assert b.shape == (8,)
        assert b.dtype == np.float32
        assert c.shape == (1,)
        assert c.dtype == np.float32

        # Randomly initialized: should not equal original fixed values
        assert not np.allclose(w, 0.75)

    def test_rebuild_from_json_and_safetensors(self):
        """Test rebuilding graph from JSON + safetensors restores exact original weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "model.json")
            st_path = os.path.join(tmpdir, "model.safetensors")
            onnx_path = os.path.join(tmpdir, "rebuilt.onnx")

            self.graph.to_json(path=json_path)
            self.graph.save_safetensors(st_path)

            # Load from JSON + Safetensors
            rebuilt = NeuronGraph.from_json(json_path, weights=st_path)

            # Check weights match exact original values
            np.testing.assert_allclose(
                rebuilt.tensor_map["conv1.weight"], self.tensor_map["conv1.weight"]
            )
            np.testing.assert_allclose(
                rebuilt.tensor_map["conv1.bias"], self.tensor_map["conv1.bias"]
            )
            np.testing.assert_allclose(
                rebuilt.tensor_map["add_const"], self.tensor_map["add_const"]
            )

            # Check nodes and attributes
            conv_node = rebuilt.nodes[0]
            assert conv_node.op_type == "Conv"
            attrs = rebuilt._backend.get_node_attrs(conv_node)
            assert attrs["kernel_shape"] == [3, 3]
            assert attrs["strides"] == [1, 1]

            # Save reconstructed ONNX model and reload
            rebuilt.save(onnx_path)
            assert os.path.exists(onnx_path)

            reloaded_onnx = NeuronGraph.load(onnx_path)
            assert len(reloaded_onnx.nodes) == 3
            np.testing.assert_allclose(
                reloaded_onnx.tensor_map["conv1.weight"],
                self.tensor_map["conv1.weight"],
            )

    def test_inplace_load_safetensors_and_replace_weights(self):
        """Test in-place weight/bias replacement like PyTorch load_state_dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_str = self.graph.to_json()
            st_path = os.path.join(tmpdir, "weights.safetensors")
            self.graph.save_safetensors(st_path)

            # Start from random weights
            rebuilt = NeuronGraph.from_json(json_str, seed=123)
            assert not np.allclose(rebuilt.tensor_map["conv1.weight"], 0.75)

            # Replace weights in-place from safetensors
            res = rebuilt.load_safetensors(st_path)
            assert res["missing_keys"] == []
            assert res["unexpected_keys"] == []

            # Check weights are now replaced
            np.testing.assert_allclose(
                rebuilt.tensor_map["conv1.weight"], self.tensor_map["conv1.weight"]
            )
            np.testing.assert_allclose(
                rebuilt.tensor_map["conv1.bias"], self.tensor_map["conv1.bias"]
            )

            # Test replace_weights with dictionary
            new_weights = {
                "conv1.weight": np.zeros((8, 3, 3, 3), dtype=np.float32),
                "conv1.bias": np.ones((8,), dtype=np.float32),
                "add_const": np.array([9.9], dtype=np.float32),
            }
            rebuilt.replace_weights(new_weights)
            np.testing.assert_allclose(
                rebuilt.tensor_map["conv1.weight"], new_weights["conv1.weight"]
            )
            np.testing.assert_allclose(
                rebuilt.tensor_map["conv1.bias"], new_weights["conv1.bias"]
            )

            # Test strict mode error checking
            with self.assertRaises(ValueError):
                rebuilt.replace_weights({"conv1.weight": np.zeros((8, 3, 3, 3))}, strict=True)

            with self.assertRaises(ValueError):
                rebuilt.replace_weights(
                    {**new_weights, "extra_layer.weight": np.zeros((2, 2))},
                    strict=True,
                )

    def test_auto_load_json_formats(self):
        """Test loading JSON via NeuronGraph constructor and load method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "model.json")
            self.graph.to_json(path=json_path)

            # Load via NeuronGraph.load
            g1 = NeuronGraph.load(json_path)
            assert len(g1.nodes) == 3

            # Load via NeuronGraph(path)
            g2 = NeuronGraph(json_path)
            assert len(g2.nodes) == 3

            # Load via NeuronGraph(json_string)
            json_str = self.graph.to_json()
            g3 = NeuronGraph(json_str)
            assert len(g3.nodes) == 3

            # Load via neuron_toolkit.from_json / load_graph_from_json
            g4 = neuron_toolkit.from_json(json_str)
            assert len(g4.nodes) == 3

    def test_query_and_rewriting_on_rebuilt_graph(self):
        """Test query, pattern matching, slicing, and rewriting on reconstructed graph."""
        json_dict = self.graph.to_dict()
        rebuilt = NeuronGraph.from_json(json_dict)

        # 1. Query
        conv_nodes = rebuilt.query().op("Conv").all()
        assert len(conv_nodes) == 1
        assert conv_nodes[0].name == "conv1"

        # 2. Pattern Matching
        from neuron_toolkit.pattern import Pattern

        pat = Pattern.op("Relu", Pattern.op("Conv"))
        matches = rebuilt.findall(pat)
        assert len(matches) == 1

        # 3. Rewriting
        rewriter = rebuilt.rewriter()
        rewriter.replace(
            [conv_nodes[0]],
            "CustomConv",
            inputs=list(conv_nodes[0].input),
            outputs=list(conv_nodes[0].output),
            name="custom_conv_1",
        )
        new_model = rewriter.build()
        new_g = NeuronGraph(new_model)
        assert len(new_g.query().op("CustomConv").all()) == 1


if __name__ == "__main__":
    unittest.main()

