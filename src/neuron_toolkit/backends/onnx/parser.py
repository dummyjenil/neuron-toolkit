"""ONNX backend parser."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import ItemsView, Iterator, KeysView, Sequence
from typing import TYPE_CHECKING, Any, cast

import onnx
from onnx import ModelProto, numpy_helper
from onnx.onnx_pb import NodeProto

from neuron_toolkit._utils import ShapeInfo, _GraphShim
from neuron_toolkit.backends.base import BaseParser, BaseRewriter
from neuron_toolkit.backends.onnx.utils import _build_shape_info
from neuron_toolkit.query import NeuronQuery

if TYPE_CHECKING:
    import numpy as np

    from neuron_toolkit.pattern import MatchResult, Pattern

log = logging.getLogger("neuron_toolkit.backends.onnx")


class LazyTensorMap(dict[str, "np.ndarray"]):
    """Lazy-loading map for ONNX initializers."""

    def __init__(self, initializers: Sequence[onnx.TensorProto]) -> None:
        super().__init__()
        self._initializers = {t.name: t for t in initializers}
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self._cache:
            return self._cache[key]
        if key in self._initializers:
            arr = numpy_helper.to_array(self._initializers[key])
            self._cache[key] = arr
            return arr
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        return (
            key in self._initializers or key in self._cache or super().__contains__(key)
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get tensor by key with optional default."""
        try:
            return self[key]
        except KeyError:
            return default

    def __len__(self) -> int:
        return len(self._initializers)

    def __iter__(self) -> Iterator[str]:
        return iter(self._initializers)

    def keys(self) -> KeysView[str]:
        return self._initializers.keys()

    def items(self) -> ItemsView[str, np.ndarray]:
        return {k: self[k] for k in self._initializers}.items()


class ONNXParser(BaseParser):
    """Load an ONNX model and expose it for querying and pattern matching."""

    def __init__(
        self,
        onnx_source: str | bytes | ModelProto,
        *,
        infer_shapes: bool = True,
        **_kwargs: object,
    ) -> None:
        """Initialize the ONNXParser with model file, bytes, or ModelProto."""
        if isinstance(onnx_source, ModelProto):
            self.model = onnx_source
        elif isinstance(onnx_source, bytes):
            self.model = onnx.load_model_from_string(onnx_source)
        else:
            log.info("Loading ONNX model from %r", onnx_source)
            self.model = onnx.load(onnx_source)

        if infer_shapes:
            try:
                self.model = onnx.shape_inference.infer_shapes(self.model)
                log.debug("Shape inference completed.")
            except Exception as exc:
                log.warning("Shape inference failed: %s", exc)

        self.nodes: list[NodeProto] = list(self.model.graph.node)
        self.tensor_map = LazyTensorMap(self.model.graph.initializer)
        self.graph_inputs: set[str] = {i.name for i in self.model.graph.input}
        self.graph_outputs: set[str] = {o.name for o in self.model.graph.output}
        self.shape_info: ShapeInfo = _build_shape_info(self.model)

        log.info(
            "Model loaded: %d nodes, %d tensors, %d inputs, "
            "%d outputs, %d shape annotations",
            len(self.nodes),
            len(self.tensor_map),
            len(self.graph_inputs),
            len(self.graph_outputs),
            len(self.shape_info),
        )

    # ------------------------------------------------------------------

    def find(self) -> NeuronQuery:
        """Return an NeuronQuery over **all** nodes in the graph."""
        return NeuronQuery(
            list(self.nodes),
            self.tensor_map,
            self.nodes,
            self.graph_inputs,
            self.graph_outputs,
            self.shape_info,
            backend=self,
        )

    def pattern_detect(
        self,
        pattern: Pattern,
        start_node: object | None = None,
        end_node: object | None = None,
    ) -> MatchResult | None:
        """Create a PatternDetector bound to this model and call match()."""
        from neuron_toolkit.pattern import PatternDetector

        shim = _GraphShim(self.nodes, self.tensor_map, self.shape_info, backend=self)
        det = PatternDetector(shim, start_node=start_node, end_node=end_node)
        return det.match(pattern)

    def rewriter(self) -> BaseRewriter:
        """Return a rewriter bound to this model."""
        from neuron_toolkit.backends.onnx.rewriter import ONNXRewriter

        return ONNXRewriter(self)

    def get_node_attrs(self, node: object) -> dict[str, object]:
        """Extract attributes from an ONNX node."""
        from neuron_toolkit.backends.onnx.utils import _node_attrs

        return _node_attrs(cast(NodeProto, node))

    def is_constant_node(self, node: object) -> bool:
        """Check if node is an ONNX Constant."""
        return getattr(node, "op_type", None) == "Constant"

    def get_constant_value(self, node: object) -> object | None:
        """Extract value from ONNX Constant node."""
        from onnx import numpy_helper

        for attr in getattr(node, "attribute", []):
            if attr.name == "value":
                return numpy_helper.to_array(attr.t)
        return None

    def summary(self) -> str:
        """Return a human-readable summary of the model."""
        op_counts = Counter(n.op_type for n in self.nodes)
        lines = [
            "ONNX model summary",
            f"  Nodes        : {len(self.nodes)}",
            f"  Tensors      : {len(self.tensor_map)}",
            f"  Graph inputs : {len(self.graph_inputs)}",
            f"  Graph outputs: {len(self.graph_outputs)}",
            f"  Shape info   : {len(self.shape_info)} annotated tensors",
            f"  Op types ({len(op_counts)}):",
        ]
        for op, cnt in op_counts.most_common():
            lines.append(f"    {op:<24} {cnt:>5}")
        return "\n".join(lines)

    def slice(
        self,
        start_points: list[str | object] | str | object,
        end_points: list[str | object] | str | object,
        output_path: str | None = None,
    ) -> ONNXParser:
        """Slice the ONNX model between start and end points and return a new ONNXParser."""
        import os
        import tempfile

        import onnx
        import onnx.utils

        from neuron_toolkit._utils import trace_subgraph_boundaries

        # 1. Trace boundaries to find inputs and outputs tensor names
        _, inputs, outputs = trace_subgraph_boundaries(
            self.nodes, self.tensor_map, start_points, end_points, self.graph_outputs
        )

        # 2. Extract using onnx.utils.extract_model via a temporary file
        temp_dir = tempfile.mkdtemp(prefix="neuron_toolkit_slice_")

        with tempfile.NamedTemporaryFile(
            suffix=".onnx", dir=temp_dir, delete=False
        ) as f_in:
            in_path = f_in.name
        with tempfile.NamedTemporaryFile(
            suffix=".onnx", dir=temp_dir, delete=False
        ) as f_out:
            out_path = f_out.name

        try:
            onnx.save(self.model, in_path)
            onnx.utils.extract_model(
                input_path=in_path,
                output_path=out_path,
                input_names=inputs,
                output_names=outputs,
                check_model=True,
                infer_shapes=True,
            )
            extracted_model = onnx.load(out_path)

            if output_path:
                onnx.save(extracted_model, output_path)

            return ONNXParser(extracted_model)
        finally:
            for p in (in_path, out_path):
                if os.path.exists(p):
                    os.remove(p)
