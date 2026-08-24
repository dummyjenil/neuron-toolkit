"""neuron_toolkit.

Unified graph analysis, querying, pattern-matching, and rewriting library.
Supports ONNX and TFLite backends.
"""

from ._types import TensorMap
from ._utils import _GraphShim
from .exporter import (
    build_onnx_model_from_json,
    compute_weight_stats,
    export_graph_dict,
    export_graph_json,
    export_safetensors,
    load_graph_from_json,
    load_safetensors,
)
from .graph import NeuronGraph
from .pattern import MatchResult, Pattern, PatternDetector
from .query import NeuronQuery
from .rewriter import NeuronRewriter

to_graph_json = export_graph_json
save_safetensors = export_safetensors
from_json = load_graph_from_json

__all__ = [
    "MatchResult",
    "NeuronGraph",
    "NeuronQuery",
    "NeuronRewriter",
    "Pattern",
    "PatternDetector",
    "TensorMap",
    "_GraphShim",
    "build_onnx_model_from_json",
    "compute_weight_stats",
    "export_graph_dict",
    "export_graph_json",
    "export_safetensors",
    "from_json",
    "load_graph_from_json",
    "load_safetensors",
    "save_safetensors",
    "to_graph_json",
]
