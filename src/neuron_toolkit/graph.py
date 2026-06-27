"""neuron_toolkit.graph.

Unified entry point for graph analysis, querying, and transformation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from neuron_toolkit._utils import _GraphShim

if TYPE_CHECKING:
    from neuron_toolkit._utils import ShapeInfo
    from neuron_toolkit.pattern import MatchResult, Pattern
    from neuron_toolkit.query import NeuronQuery
    from neuron_toolkit.rewriter import NeuronRewriter

log = logging.getLogger("neuron_toolkit.graph")


class NeuronGraph:
    """Unified interface for a model graph (ONNX or TFLite)."""

    def __init__(self, source: object, **kwargs: Any) -> None:
        self._backend: Any
        if isinstance(source, str):
            # Auto-load if it's a path
            temp_g = self.load(source, **kwargs)
            self._backend = temp_g._backend  # noqa: SLF001
        elif isinstance(source, bytes):
            # Check flatbuffer magic header for TFLite at offset 4
            if len(source) >= 8 and source[4:8] == b"TFL3":  # noqa: PLR2004
                from neuron_toolkit.backends.tflite.parser import (  # noqa: PLC0415
                    TFLiteParser,
                )

                self._backend = TFLiteParser(source, **kwargs)
            else:
                from neuron_toolkit.backends.onnx.parser import (  # noqa: PLC0415
                    ONNXParser,
                )

                self._backend = ONNXParser(cast(Any, source), **kwargs)
        elif hasattr(source, "graph") and hasattr(
            source.graph, "node"
        ):  # ONNX ModelProto
            from neuron_toolkit.backends.onnx.parser import ONNXParser  # noqa: PLC0415

            self._backend = ONNXParser(cast(Any, source), **kwargs)
        elif hasattr(source, "Subgraphs") and hasattr(
            source, "OperatorCodes"
        ):  # TFLite Model
            from neuron_toolkit.backends.tflite.parser import (  # noqa: PLC0415
                TFLiteParser,
            )

            self._backend = TFLiteParser(source, **kwargs)
        else:
            # Assume it's already a backend parser
            self._backend = source

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> NeuronGraph:
        """Load a model from path and return a NeuronGraph."""
        from pathlib import Path  # noqa: PLC0415

        ext = Path(path).suffix.lower()
        if ext == ".onnx":
            from neuron_toolkit.backends.onnx.parser import ONNXParser  # noqa: PLC0415

            return cls(ONNXParser(path, **kwargs))
        if ext in (".tflite", ".bin"):
            from neuron_toolkit.backends.tflite.parser import (  # noqa: PLC0415
                TFLiteParser,
            )

            return cls(TFLiteParser(path, **kwargs))

        msg = f"Unsupported model format: {ext}"
        raise ValueError(msg)

    @property
    def nodes(self) -> list[object]:
        """Return the list of nodes in the graph."""
        return getattr(self._backend, "nodes", [])

    @property
    def tensor_map(self) -> dict[str, object]:
        """Return the dictionary of initializers/parameter tensors."""
        return getattr(self._backend, "tensor_map", {})

    @property
    def shape_info(self) -> ShapeInfo:
        """Return shape and dtype information for each tensor."""
        return getattr(self._backend, "shape_info", {})

    def query(self) -> NeuronQuery:
        """Alias for find() — return a query over all nodes."""
        return self._backend.find()

    def match(self, pattern: Pattern) -> MatchResult | None:
        """Find the first occurrence of *pattern* in the graph."""
        results = self.findall(pattern)
        return results[0] if results else None

    def findall(self, pattern: Pattern) -> list[MatchResult]:
        """Find all occurrences of *pattern* in the graph."""
        from neuron_toolkit.pattern import PatternDetector  # noqa: PLC0415

        det = PatternDetector(self._shim())
        return det.find_all(pattern)

    def replace(
        self,
        pattern: Pattern,
        new_op: str,
        name: str | None = None,
        **attrs: object,
    ) -> NeuronRewriter:
        """Replace all matches of *pattern* with a new operator.

        This is a fluent shortcut for finding all matches and applying a rewrite.
        """
        rewriter = self.rewriter()
        matches = self.findall(pattern)
        for m in matches:
            rewriter.replace_from_result(m, new_op, name=name, **attrs)
        return rewriter

    def rewriter(self) -> NeuronRewriter:
        """Return a GraphRewriter bound to this model."""
        from neuron_toolkit.rewriter import NeuronRewriter  # noqa: PLC0415

        return NeuronRewriter(self._backend.rewriter())

    def _shim(self) -> _GraphShim:
        return _GraphShim(
            self.nodes, self.tensor_map, self.shape_info, backend=self._backend
        )

    @property
    def passes(self) -> GraphPasses:
        """Return a namespace for common graph optimization passes."""
        return GraphPasses(self)


class GraphPasses:
    """Namespace for common graph optimization passes."""

    def __init__(self, graph: NeuronGraph) -> None:
        self.graph = graph

    def fuse_conv_bn(self) -> NeuronRewriter:
        """Example pass: Fuse Conv and BatchNormalization."""
        from neuron_toolkit.pattern import Pattern  # noqa: PLC0415

        conv = Pattern.op("Conv").capture("conv")
        bn = Pattern.op("BatchNormalization", conv).capture("bn")
        return self.graph.replace(bn, "Conv")  # Placeholder for real fusion logic
