"""
onnx_toolkit.rewriter
======================
GraphRewriter — modify an ONNX graph by replacing matched subgraphs with
new nodes, removing nodes, or inserting nodes.

Usage
-----
    from onnx_toolkit import ONNXParser, Pattern, GraphRewriter

    parser   = ONNXParser("model.onnx")
    rewriter = GraphRewriter(parser)

    x = Pattern.any().capture("x")
    for result in parser.find().match_results(Pattern.gelu(x)):
        rewriter.replace(
            nodes    = result.nodes,
            new_op   = "Gelu",
            inputs   = [result.bindings["x"].output[0]],
            outputs  = [result.start.output[0]],
        )

    rewriter.build("model_fused.onnx")

All mutations are staged — the original model is untouched until build().
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import onnx
from onnx import helper, numpy_helper, TensorProto
from onnx.onnx_pb import NodeProto

from .pattern import MatchResult

log = logging.getLogger("onnx_toolkit.rewriter")


class GraphRewriter:
    """
    Staged graph rewriter.  Accumulates a set of edits and applies them
    all at once when :meth:`build` is called.

    Edits are expressed as:
    - *replacements*: a set of nodes to remove + one new node to insert
    - *insertions*:   a single new node added before a given node
    - *deletions*:    a set of nodes to remove with no replacement

    Parameters
    ----------
    parser : ONNXParser
        The model to rewrite.  The original model is never modified.
    """

    def __init__(self, parser: "ONNXParser") -> None:  # type: ignore[name-defined]
        from .parser import ONNXParser
        if not isinstance(parser, ONNXParser):
            raise TypeError("GraphRewriter requires an ONNXParser instance")
        self._parser    = parser
        self._to_remove: List[str]      = []   # node names to drop
        self._to_insert: List[NodeProto] = []   # new nodes to add

    # ------------------------------------------------------------------
    # Edit API
    # ------------------------------------------------------------------

    def replace(
        self,
        nodes:   Sequence[NodeProto],
        new_op:  str,
        inputs:  List[str],
        outputs: List[str],
        name:    Optional[str] = None,
        **attrs: object,
    ) -> "GraphRewriter":
        """
        Replace *nodes* with a single new node of type *new_op*.

        Parameters
        ----------
        nodes   : the nodes to remove (typically result.nodes from a MatchResult)
        new_op  : op_type of the replacement node
        inputs  : input tensor names for the new node
        outputs : output tensor names for the new node
        name    : optional name for the new node (auto-generated if omitted)
        **attrs : keyword attributes passed to onnx.helper.make_node

        Returns self for chaining.
        """
        for n in nodes:
            if n.name not in self._to_remove:
                self._to_remove.append(n.name)
        node_name = name or f"{new_op}_{len(self._to_insert)}_rewrite"
        new_node  = helper.make_node(new_op, inputs=inputs, outputs=outputs,
                                     name=node_name, **attrs)
        self._to_insert.append(new_node)
        log.debug("replace(): scheduled removal of %d node(s), inserting %r",
                  len(nodes), node_name)
        return self

    def replace_from_result(
        self,
        result:  MatchResult,
        new_op:  str,
        inputs:  Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        name:    Optional[str] = None,
        **attrs: object,
    ) -> "GraphRewriter":
        """
        Convenience wrapper: replace all nodes in *result* with a new op.

        *inputs* defaults to the first output of result.end (the boundary node).
        *outputs* defaults to the first output of result.start.
        """
        ins  = inputs  or [result.end.output[0]]
        outs = outputs or [result.start.output[0]]
        return self.replace(result.nodes, new_op, ins, outs, name=name, **attrs)

    def delete(self, nodes: Sequence[NodeProto]) -> "GraphRewriter":
        """
        Remove *nodes* from the graph entirely.

        Warning: this will produce a disconnected graph unless the removed
        nodes' consumers are also removed or their inputs are rewired.
        Returns self for chaining.
        """
        for n in nodes:
            if n.name not in self._to_remove:
                self._to_remove.append(n.name)
        log.debug("delete(): scheduled removal of %d node(s)", len(nodes))
        return self

    def insert_before(
        self,
        target_node: NodeProto,
        new_op:      str,
        inputs:      List[str],
        outputs:     List[str],
        name:        Optional[str] = None,
        **attrs:     object,
    ) -> "GraphRewriter":
        """
        Insert a new node whose outputs feed *target_node*.

        The caller is responsible for ensuring *outputs* matches the inputs
        that *target_node* expects.  Returns self for chaining.
        """
        node_name = name or f"{new_op}_{len(self._to_insert)}_insert"
        new_node  = helper.make_node(new_op, inputs=inputs, outputs=outputs,
                                     name=node_name, **attrs)
        self._to_insert.append(new_node)
        log.debug("insert_before(%r): scheduled insertion of %r", target_node.name, node_name)
        return self

    def reset(self) -> "GraphRewriter":
        """Discard all pending edits."""
        self._to_remove.clear()
        self._to_insert.clear()
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, output_path: Optional[str] = None) -> onnx.ModelProto:
        """
        Apply all staged edits and return the new ModelProto.

        If *output_path* is given the model is also saved to that path.

        The edit order is:
        1. Remove all scheduled nodes.
        2. Append inserted nodes at the end of the node list.
           (The graph is re-sorted topologically before saving so ordering
            is always valid.)

        Raises
        ------
        ValueError  if no edits have been staged.
        """
        if not self._to_remove and not self._to_insert:
            raise ValueError("No edits staged.  Call replace(), delete(), or insert_before() first.")

        remove_set = set(self._to_remove)
        orig_graph = self._parser.model.graph

        # Keep nodes not scheduled for removal
        kept_nodes = [n for n in orig_graph.node if n.name not in remove_set]
        # Append new nodes
        all_nodes  = kept_nodes + self._to_insert

        # Re-use original initializers, inputs, outputs
        new_graph = helper.make_graph(
            nodes       = all_nodes,
            name        = orig_graph.name or "rewritten",
            inputs      = list(orig_graph.input),
            outputs     = list(orig_graph.output),
            initializer = list(orig_graph.initializer),
        )

        # Copy model metadata
        new_model = helper.make_model(new_graph)
        new_model.ir_version   = self._parser.model.ir_version
        new_model.opset_import.extend(self._parser.model.opset_import)

        # Run shape inference on the result
        try:
            new_model = onnx.shape_inference.infer_shapes(new_model)
        except Exception as exc:
            log.warning("Shape inference failed after rewrite: %s", exc)

        if output_path:
            onnx.save(new_model, output_path)
            log.info("Rewritten model saved to %r", output_path)

        log.debug(
            "build(): removed %d node(s), inserted %d node(s), result has %d node(s)",
            len(self._to_remove), len(self._to_insert), len(all_nodes),
        )
        return new_model
