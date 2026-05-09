"""onnx_toolkit.

Unified ONNX parsing, querying, pattern-matching, and graph-rewriting library.
"""

from ._types import TensorMap
from ._utils import _GraphShim
from .graph import ONNXGraph
from .parser import ONNXParser
from .pattern import MatchResult, Pattern, PatternDetector
from .query import ONNXQuery
from .rewriter import GraphRewriter

__all__ = [
    "GraphRewriter",
    "MatchResult",
    "ONNXGraph",
    "ONNXParser",
    "ONNXQuery",
    "Pattern",
    "PatternDetector",
    "TensorMap",
    "_GraphShim",
]
