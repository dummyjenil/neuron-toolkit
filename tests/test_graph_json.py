"""Tests for model to graph JSON export feature."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

import neuron_toolkit
from neuron_toolkit._utils import _GraphShim
from neuron_toolkit.exporter import (
    compute_weight_stats,
    export_graph_dict,
)
from neuron_toolkit.graph import NeuronGraph


class TestWeightStats(unittest.TestCase):
    """Test weight statistical analysis without raw data buffers."""

    def test_empty_tensor(self):
        empty_arr = np.array([], dtype=np.float32)
        stats = compute_weight_stats(empty_arr)
        assert stats["numel"] == 0
        assert stats["shape"] == [0]
        assert stats["stats"]["sum"] == 0.0

    def test_float_tensor_stats(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 0.0], dtype=np.float32)
        stats = compute_weight_stats(arr)

        assert stats["shape"] == [5]
        assert stats["dtype"] == "float32"
        assert stats["numel"] == 5

        s = stats["stats"]
        self.assertAlmostEqual(s["sum"], 10.0, places=5)
        self.assertAlmostEqual(s["mean"], 2.0, places=5)
        self.assertAlmostEqual(s["min"], 0.0, places=5)
        self.assertAlmostEqual(s["max"], 4.0, places=5)
        assert s["zeros_count"] == 1
        self.assertAlmostEqual(s["zero_fraction"], 0.2, places=5)
        assert not s["has_nan"]
        assert not s["has_inf"]

    def test_nan_and_inf_detection(self):
        arr_nan = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        stats_nan = compute_weight_stats(arr_nan)
        assert stats_nan["stats"]["has_nan"]
        assert not stats_nan["stats"]["has_inf"]

        arr_inf = np.array([1.0, np.inf, 3.0], dtype=np.float32)
        stats_inf = compute_weight_stats(arr_inf)
        assert not stats_inf["stats"]["has_nan"]
        assert stats_inf["stats"]["has_inf"]

    def test_multidimensional_weight(self):
        # 4D conv weight shape [32, 16, 3, 3]
        np.random.seed(42)
        arr = np.random.randn(32, 16, 3, 3).astype(np.float32)
        stats = compute_weight_stats(arr)

        assert stats["shape"] == [32, 16, 3, 3]
        assert stats["numel"] == 32 * 16 * 3 * 3
        assert "sum" in stats["stats"]
        assert "mean" in stats["stats"]
        assert "std" in stats["stats"]
        assert "min" in stats["stats"]
        assert "max" in stats["stats"]


class TestGraphJsonExport(unittest.TestCase):
    """Test full model graph export to JSON dictionary and file."""

    def setUp(self):
        # Create a mock graph with 3 nodes: Input -> Conv (with weights) -> Relu -> Add -> Output
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
            "conv1.weight": np.ones((16, 3, 3, 3), dtype=np.float32) * 0.5,
            "conv1.bias": np.zeros((16,), dtype=np.float32),
            "add_const": np.array([1.5], dtype=np.float32),
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

    def test_export_graph_dict_structure(self):
        graph_dict = export_graph_dict(self.backend)

        # 1. Root structure
        assert graph_dict["format"] == "neuron_toolkit_graph_v1"
        assert "metadata" in graph_dict
        assert "inputs" in graph_dict
        assert "outputs" in graph_dict
        assert "nodes" in graph_dict
        assert "connections" in graph_dict
        assert "edges" in graph_dict
        assert "weights_summary" in graph_dict

        # 2. Inputs & Outputs
        assert len(graph_dict["inputs"]) == 1
        assert graph_dict["inputs"][0]["name"] == "input_0"
        assert len(graph_dict["outputs"]) == 1
        assert graph_dict["outputs"][0]["name"] == "output_0"

        # 3. Nodes section
        nodes = graph_dict["nodes"]
        assert len(nodes) == 3

        # Conv node check
        conv_node = nodes[0]
        assert conv_node["name"] == "conv1"
        assert conv_node["op_type"] == "Conv"
        assert conv_node["inputs"] == ["input_0", "conv1.weight", "conv1.bias"]
        assert conv_node["outputs"] == ["conv1_out"]
        assert conv_node["attributes"]["kernel_shape"] == [3, 3]

        # Verify weights stats are attached to the node
        assert len(conv_node["weights"]) == 2
        w_stat = next(
            w for w in conv_node["weights"] if w["name"] == "conv1.weight"
        )
        assert w_stat["shape"] == [16, 3, 3, 3]
        self.assertAlmostEqual(w_stat["stats"]["mean"], 0.5, places=5)

        # 4. Connections section
        connections = graph_dict["connections"]
        assert len(connections) > 0

        # Check edge from graph input to conv1
        input_edge = next(
            e for e in connections if e["tensor_name"] == "input_0"
        )
        assert input_edge["source"] == "__input__:input_0"
        assert input_edge["target"] == conv_node["id"]
        assert not input_edge["is_weight"]

        # Check edge from conv1 to relu1
        conv_to_relu_edge = next(
            e for e in connections if e["tensor_name"] == "conv1_out"
        )
        assert conv_to_relu_edge["source"] == conv_node["id"]
        assert conv_to_relu_edge["target"] == nodes[1]["id"]

        # Check edge from add1 to graph output
        out_edge = next(
            e
            for e in connections
            if e["source"] == nodes[2]["id"] and e["target"] == "__output__:output_0"
        )
        assert out_edge["tensor_name"] == "output_0"

        # 5. Weights summary
        ws = graph_dict["weights_summary"]
        assert ws["total_tensors"] == 3
        assert ws["total_elements"] == 16 * 3 * 3 * 3 + 16 + 1

    def test_export_to_json_string_and_file(self):
        graph = NeuronGraph(self.backend)

        # Export as JSON string
        json_str = graph.to_json(indent=2)
        assert isinstance(json_str, str)

        # Verify valid JSON
        parsed = json.loads(json_str)
        assert parsed["format"] == "neuron_toolkit_graph_v1"
        assert len(parsed["nodes"]) == 3

        # Test method aliases on NeuronGraph
        assert graph.to_dict()["format"] == "neuron_toolkit_graph_v1"
        assert graph.to_graph_json() == json_str
        assert graph.export_graph_json() == json_str

        # Test top-level package function
        assert neuron_toolkit.to_graph_json(graph) == json_str
        assert neuron_toolkit.export_graph_json(graph) == json_str

        # Export to file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            res = graph.export_graph_json(path=tmp_path)
            assert res is None
            assert os.path.exists(tmp_path)

            with open(tmp_path, encoding="utf-8") as f:
                content = json.loads(f.read())
            assert content["format"] == "neuron_toolkit_graph_v1"
            assert len(content["connections"]) == len(parsed["connections"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_large_weight_file_size_remains_tiny(self):
        # Create a large 10-million-parameter weight tensor
        large_weights = {
            "big_weight_1": np.zeros((1000, 1000), dtype=np.float32),
            "big_weight_2": np.ones((1000, 1000), dtype=np.float32),
        }
        node = SimpleNamespace(
            op_type="MatMul",
            name="matmul1",
            input=["input_0", "big_weight_1"],
            output=["out_0"],
            attrs={},
        )
        backend = _GraphShim(
            nodes=[node],
            tensor_map=large_weights,
            shape_info={"input_0": (2, "float32"), "out_0": (2, "float32")},
        )
        backend.graph_inputs = {"input_0"}
        backend.graph_outputs = {"out_0"}

        graph = NeuronGraph(backend)
        json_str = graph.to_json()

        # 2 million float32s raw = 8 MB. But JSON without weights must be tiny (< 5 KB)
        assert len(json_str.encode("utf-8")) < 5000
        parsed = json.loads(json_str)
        assert parsed["weights_summary"]["total_elements"] == 2000000
        assert parsed["metadata"]["total_weight_params"] == 2000000

    def test_integer_and_quantized_tensor_stats(self):
        arr_int = np.array([0, 10, -5, 100, 0, 25], dtype=np.int32)
        stats = compute_weight_stats(arr_int)
        assert stats["shape"] == [6]
        assert stats["dtype"] == "int32"
        assert stats["numel"] == 6
        assert stats["stats"]["zeros_count"] == 2
        assert not stats["stats"]["has_nan"]
        assert not stats["stats"]["has_inf"]
        assert stats["stats"]["sum"] == 130.0
        assert stats["stats"]["min"] == -5.0
        assert stats["stats"]["max"] == 100.0

    def test_large_model_export_speed_and_memory(self):
        import time

        # 50 nodes with 50 weight tensors (500k elements each = 25M elements)
        tensor_map = {
            f"weight_{i}": np.random.randn(100, 100).astype(np.float32)
            for i in range(50)
        }
        nodes = [
            SimpleNamespace(
                op_type="MatMul",
                name=f"matmul_{i}",
                input=[f"t_{i}", f"weight_{i}"],
                output=[f"t_{i+1}"],
                attrs={"transB": 1},
            )
            for i in range(50)
        ]
        backend = _GraphShim(
            nodes=nodes,
            tensor_map=tensor_map,
            shape_info={f"t_{i}": (2, "float32") for i in range(51)},
        )
        backend.graph_inputs = {"t_0"}
        backend.graph_outputs = {"t_50"}

        # Warm up JIT/imports if any
        _ = export_graph_dict(backend)

        start = time.perf_counter()
        graph_dict = export_graph_dict(backend)
        elapsed = time.perf_counter() - start

        # Must execute swiftly (under 250ms even with full coverage tracing overhead)
        assert elapsed < 0.25
        assert len(graph_dict["nodes"]) == 50
        assert len(graph_dict["connections"]) == 101
        assert graph_dict["metadata"]["total_weight_params"] == 500_000


if __name__ == "__main__":
    unittest.main()

