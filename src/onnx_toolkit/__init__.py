"""
onnx_toolkit
=============
Unified ONNX parsing and pattern-matching library.

Provides two tightly integrated components:

  ONNXParser / ONNXQuery
      Load a model and query its graph with a fluent, chainable API.

  Pattern / PatternDetector
      Describe structural subgraph patterns and match them against the graph
      using DFS with commutativity support.

The two components are coupled: PatternDetector can be seeded directly from
an ONNXQuery result, and ONNXQuery exposes a ``.matches(pattern)`` convenience
method that delegates to PatternDetector internally.

Quick start
-----------
>>> from onnx_toolkit import ONNXParser, Pattern
>>>
>>> parser = ONNXParser("model.onnx")
>>>
>>> # Query API
>>> convs = parser.find().find_by_op_type("Conv").has_params()
>>> print(convs)
>>>
>>> # Pattern matching
>>> x   = Pattern.any()
>>> end = parser.pattern_detect(
...     Pattern.gelu(x),
...     start_node="Mul_42",
... )
>>> if end:
...     print("GELU pattern terminates at:", end.name)

Logging
-------
Uses Python's standard ``logging`` library.  To enable debug output::

    import logging
    logging.basicConfig(level=logging.DEBUG)

Sub-loggers
-----------
  onnx_toolkit          – top-level / parser messages
  onnx_toolkit.query    – ONNXQuery filter & traversal steps
  onnx_toolkit.pattern  – Pattern construction messages
  onnx_toolkit.detect   – PatternDetector DFS walk & decisions
"""

from .parser import ONNXParser
from .pattern import Pattern, PatternDetector
from .query import ONNXQuery
from ._types import TensorMap
from ._utils import _GraphShim

__all__ = [
    "ONNXParser",
    "ONNXQuery",
    "Pattern",
    "PatternDetector",
    "TensorMap",
    # _GraphShim is semi-internal but exported for advanced users who build
    # their own detectors without a full ModelProto.
    "_GraphShim",
]
