"""neuron_toolkit.

Unified graph analysis, querying, pattern-matching, and rewriting library.
Supports ONNX and TFLite backends.
"""

from ._types import TensorMap
from ._utils import _GraphShim
from .exporter import (
    compute_weight_stats,
    export_graph_dict,
    export_graph_json,
)
from .graph import NeuronGraph
from .pattern import MatchResult, Pattern, PatternDetector
from .query import NeuronQuery
from .rewriter import NeuronRewriter

to_graph_json = export_graph_json

__all__ = [
    "MatchResult",
    "NeuronGraph",
    "NeuronQuery",
    "NeuronRewriter",
    "Pattern",
    "PatternDetector",
    "TensorMap",
    "_GraphShim",
    "compute_weight_stats",
    "export_graph_dict",
    "export_graph_json",
    "to_graph_json",
]
