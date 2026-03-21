"""
onnx_toolkit
=============
Unified ONNX parsing, querying, pattern-matching, and graph-rewriting library.

Components
----------
ONNXParser
    Load a model.  Entry point for everything else.

ONNXQuery
    Lazy, chainable view over graph nodes.  Supports filters, traversal,
    topological sort, apply(), pattern matching, and export to DOT /
    DataFrame / .onnx.

Pattern
    DSL for structural subgraph patterns.  Supports captures, attribute
    constraints, shape / dtype constraints, union patterns, and a full
    set of activation-function helpers.

MatchResult
    Returned by PatternDetector.match() and ONNXQuery.match_results().
    Carries start node, end node, all visited nodes, and named bindings.

PatternDetector
    Low-level engine.  Use PatternDetector(model, start_node).match(pat)
    or PatternDetector(model).find_all(pat) directly when you need full
    control.

GraphRewriter
    Staged graph mutation.  Stage replace / delete / insert_before edits,
    then call build() to produce a rewritten ModelProto.

Quick start
-----------
    from onnx_toolkit import ONNXParser, Pattern, GraphRewriter

    parser = ONNXParser("model.onnx")

    # Query
    convs = parser.find().find_by_op_type("Conv").has_params()
    print(convs)

    # Pattern with captures + attribute constraint
    x = Pattern.any().capture("x")
    depthwise_conv = Pattern.op("Conv").where(group=lambda g: g > 1)

    # Find and inspect matches
    for r in parser.find().match_results(depthwise_conv):
        print(r.start.name, r.bindings)

    # Union pattern
    act = Pattern.any_of(Pattern.relu(x), Pattern.gelu(x))
    print(parser.find().matches(act))

    # Rewrite: fuse GELU subgraphs
    rw = parser.rewriter()
    for r in parser.find().match_results(Pattern.gelu(x)):
        rw.replace_from_result(r, new_op="Gelu")
    rw.build("model_fused.onnx")

    # Export subgraph
    parser.find().find_by_op_type("Conv").ancestors().to_onnx("conv_sub.onnx")
    dot = parser.find().to_dot()

Logging
-------
    import logging
    logging.basicConfig(level=logging.DEBUG)

Sub-loggers: onnx_toolkit, onnx_toolkit.query, onnx_toolkit.pattern,
             onnx_toolkit.detect, onnx_toolkit.rewriter
"""

from .parser    import ONNXParser
from .pattern   import Pattern, PatternDetector, MatchResult
from .query     import ONNXQuery
from .rewriter  import GraphRewriter
from ._types    import TensorMap
from ._utils    import _GraphShim

__all__ = [
    "ONNXParser",
    "ONNXQuery",
    "Pattern",
    "PatternDetector",
    "MatchResult",
    "GraphRewriter",
    "TensorMap",
    "_GraphShim",
]
