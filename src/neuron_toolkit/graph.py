"""neuron_toolkit.graph.

Unified entry point for graph analysis, querying, and transformation.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, cast

from neuron_toolkit._utils import _GraphShim

if TYPE_CHECKING:
    from neuron_toolkit._utils import ShapeInfo
    from neuron_toolkit.pattern import MatchResult, Pattern
    from neuron_toolkit.query import NeuronQuery
    from neuron_toolkit.rewriter import NeuronRewriter

log = logging.getLogger("neuron_toolkit.graph")


def _is_tflite_runnable() -> bool:
    """Check if LiteRT can be loaded and run without crashing."""
    import subprocess
    import sys

    try:
        res = subprocess.run(
            [
                sys.executable,
                "-c",
                "from ai_edge_litert.interpreter import Interpreter",
            ],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return False
    else:
        return res.returncode == 0


class NeuronGraph:
    """Unified interface for a model graph (ONNX or TFLite)."""

    def __init__(self, source: object, **kwargs: Any) -> None:
        self._backend: Any
        if isinstance(source, str):
            # Auto-load if it's a path
            temp_g = self.load(source, **kwargs)
            self._backend = temp_g._backend
        elif isinstance(source, bytes):
            # Check flatbuffer magic header for TFLite at offset 4
            if len(source) >= 8 and source[4:8] == b"TFL3":
                from neuron_toolkit.backends.tflite.parser import (
                    TFLiteParser,
                )

                self._backend = TFLiteParser(source, **kwargs)
            else:
                from neuron_toolkit.backends.onnx.parser import (
                    ONNXParser,
                )

                self._backend = ONNXParser(cast(Any, source), **kwargs)
        elif hasattr(source, "graph") and hasattr(
            source.graph, "node"
        ):  # ONNX ModelProto
            from neuron_toolkit.backends.onnx.parser import ONNXParser

            self._backend = ONNXParser(cast(Any, source), **kwargs)
        elif hasattr(source, "Subgraphs") and hasattr(
            source, "OperatorCodes"
        ):  # TFLite Model
            from neuron_toolkit.backends.tflite.parser import (
                TFLiteParser,
            )

            self._backend = TFLiteParser(source, **kwargs)
        else:
            # Assume it's already a backend parser
            self._backend = source

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> NeuronGraph:
        """Load a model from path and return a NeuronGraph."""
        from pathlib import Path

        ext = Path(path).suffix.lower()
        if ext == ".onnx":
            from neuron_toolkit.backends.onnx.parser import ONNXParser

            return cls(ONNXParser(path, **kwargs))
        if ext in (".tflite", ".bin"):
            from neuron_toolkit.backends.tflite.parser import (
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
        from neuron_toolkit.pattern import PatternDetector

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
        from neuron_toolkit.rewriter import NeuronRewriter

        return NeuronRewriter(self._backend.rewriter())

    def _shim(self) -> _GraphShim:
        return _GraphShim(
            self.nodes, self.tensor_map, self.shape_info, backend=self._backend
        )

    def slice(
        self,
        start_points: list[str | object] | str | object,
        end_points: list[str | object] | str | object,
        output_path: str | None = None,
    ) -> NeuronGraph:
        """Slice the model between start and end points and return a new NeuronGraph."""
        sliced_backend = self._backend.slice(start_points, end_points, output_path)
        return NeuronGraph(sliced_backend)

    def slice_node(
        self, node: str | object, output_path: str | None = None
    ) -> NeuronGraph:
        """Extract a single node as a new model and return a new NeuronGraph."""
        return self.slice(node, node, output_path)

    def save(self, path: str) -> None:
        """Save the current model to the specified path."""
        if hasattr(self._backend, "model"):  # ONNX
            import onnx

            onnx.save(self._backend.model, path)
        elif hasattr(self._backend, "source"):  # TFLite
            source = self._backend.source
            if isinstance(source, bytes):
                from pathlib import Path

                Path(path).write_bytes(source)
            elif isinstance(source, str):
                import shutil

                shutil.copy(source, path)
            else:
                msg = "Cannot save this model type directly. Source is a parsed object."
                raise ValueError(msg)
        else:
            msg = "Unsupported backend type for saving."
            raise ValueError(msg)

    def get_weights(self) -> dict[str, Any]:
        """Return a dictionary of all parameter/initializer weight tensors in the graph."""
        return dict(self.tensor_map)

    def to_dict(self) -> dict[str, Any]:
        """Export the model graph structure as a Python dictionary without raw weights."""
        from neuron_toolkit.exporter import export_graph_dict

        return export_graph_dict(self)

    def to_json(self, path: str | None = None, indent: int = 2) -> str | None:
        """Export the model graph structure as a JSON string or save to a file path."""
        from neuron_toolkit.exporter import export_graph_json

        return export_graph_json(self, path=path, indent=indent)

    def export_graph_json(self, path: str | None = None, indent: int = 2) -> str | None:
        """Export the model graph structure to JSON format (alias for to_json)."""
        return self.to_json(path=path, indent=indent)

    def to_graph_json(self, path: str | None = None, indent: int = 2) -> str | None:
        """Export the model graph structure to JSON format (alias for to_json)."""
        return self.to_json(path=path, indent=indent)

    def compare_outputs(
        self,
        start_points: list[str | object] | str | object,
        end_points: list[str | object] | str | object,
        pytorch_module: Any,
        inputs_dict: dict[str, Any],
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> dict[str, Any]:
        """Execute the sliced model and PyTorch module on inputs and compare outputs."""
        import numpy as np
        import torch

        # 1. Run the PyTorch module
        with torch.no_grad():
            torch_inputs = []
            if len(inputs_dict) == 1:
                val = next(iter(inputs_dict.values()))
                torch_inputs.append(torch.from_numpy(np.asarray(val)))
            else:
                import inspect

                callable_module = getattr(pytorch_module, "forward", pytorch_module)
                params = list(inspect.signature(callable_module).parameters.keys())
                unused = dict(inputs_dict)
                # Pair arguments by name whenever possible…
                torch_inputs.extend(
                    torch.from_numpy(np.asarray(unused.pop(param)))
                    for param in params
                    if param in unused
                )
                # …and fill any remaining parameters with the leftover values
                # in insertion order so positionally-passed tensors line up.
                while len(torch_inputs) < len(params) and unused:
                    key = next(iter(unused))
                    val = unused.pop(key)
                    torch_inputs.append(torch.from_numpy(np.asarray(val)))

            pytorch_outputs = pytorch_module(*torch_inputs)

        # Normalize PyTorch outputs to numpy arrays
        if isinstance(pytorch_outputs, torch.Tensor):
            pytorch_outputs_np = [pytorch_outputs.cpu().numpy()]
        elif isinstance(pytorch_outputs, tuple):
            pytorch_outputs_np = [
                t.cpu().numpy() for t in pytorch_outputs if isinstance(t, torch.Tensor)
            ]
        elif isinstance(pytorch_outputs, dict):
            pytorch_outputs_np = {
                k: v.cpu().numpy()
                for k, v in pytorch_outputs.items()
                if isinstance(v, torch.Tensor)
            }
        else:
            pytorch_outputs_np = [pytorch_outputs]

        # 2. Run the sliced model
        sliced_graph = self.slice(start_points, end_points)
        if hasattr(sliced_graph._backend, "model"):  # ONNX
            import onnxruntime as ort

            model_bytes = sliced_graph._backend.model.SerializeToString()
            session = ort.InferenceSession(model_bytes)

            run_inputs = {}
            session_inputs = [i.name for i in session.get_inputs()]
            used_keys: set[str] = set()
            for name in session_inputs:
                if name in inputs_dict:
                    run_inputs[name] = inputs_dict[name]
                    used_keys.add(name)
                    continue
                matched = False
                for k, v in inputs_dict.items():
                    if k in used_keys:
                        continue
                    if k in name or name in k:
                        run_inputs[name] = v
                        matched = True
                        used_keys.add(k)
                        break
                if not matched and len(inputs_dict) == 1:
                    run_inputs[name] = next(iter(inputs_dict.values()))
                elif not matched and len(inputs_dict) == len(session_inputs):
                    # Positional fallback: match up remaining inputs by order.
                    for k, v in inputs_dict.items():
                        if k not in used_keys:
                            run_inputs[name] = v
                            used_keys.add(k)
                            matched = True
                            break

            output_names = [o.name for o in session.get_outputs()]
            outputs_list = session.run(output_names, run_inputs)
            model_outputs = dict(zip(output_names, outputs_list, strict=False))
        else:  # TFLite
            if not _is_tflite_runnable():
                msg = (
                    "LiteRT interpreter is not usable on this CPU "
                    "due to instruction set mismatch (AVX/AVX2 support missing)."
                )
                raise RuntimeError(msg)
            from ai_edge_litert.interpreter import Interpreter

            model_bytes = sliced_graph._backend.source
            if not isinstance(model_bytes, bytes):
                msg = "Sliced TFLite model source must be bytes."
                raise ValueError(msg)

            interpreter = Interpreter(model_content=model_bytes)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            tflite_used_keys: set[str] = set()
            for detail in input_details:
                name = detail["name"]
                if name in inputs_dict:
                    interpreter.set_tensor(detail["index"], inputs_dict[name])
                    tflite_used_keys.add(name)
                    continue
                matched = False
                for k, v in inputs_dict.items():
                    if k in tflite_used_keys:
                        continue
                    if k in name or name in k:
                        interpreter.set_tensor(detail["index"], v)
                        matched = True
                        tflite_used_keys.add(k)
                        break
                if not matched and len(inputs_dict) == 1:
                    interpreter.set_tensor(
                        detail["index"], next(iter(inputs_dict.values()))
                    )
                elif not matched and len(inputs_dict) == len(input_details):
                    # Positional fallback: match up remaining inputs by order.
                    for k, v in inputs_dict.items():
                        if k not in tflite_used_keys:
                            interpreter.set_tensor(detail["index"], v)
                            tflite_used_keys.add(k)
                            matched = True
                            break

            interpreter.invoke()

            model_outputs = {}
            for detail in output_details:
                model_outputs[detail["name"]] = interpreter.get_tensor(detail["index"])

        # 3. Match and compare
        # TFLite/ONNX slicing may reorder graph outputs, so re-align the model
        # outputs to the caller-declared endpoint order before pairing.
        declared_endpoints = (
            [end_points]
            if isinstance(end_points, str)
            else (
                list(end_points)
                if isinstance(end_points, (list, tuple))
                else [end_points]
            )
        )
        if declared_endpoints and all(isinstance(e, str) for e in declared_endpoints):
            model_outs_list = [
                model_outputs[e] for e in declared_endpoints if e in model_outputs
            ]
        else:
            model_outs_list = list(model_outputs.values())
        matches = []
        if isinstance(pytorch_outputs_np, dict):
            for k, py_val in pytorch_outputs_np.items():
                model_val = None
                for mk, mv in model_outputs.items():
                    if k in mk or mk in k:
                        model_val = mv
                        break
                if model_val is None and len(model_outputs) == 1:
                    model_val = next(iter(model_outputs.values()))

                if model_val is not None:
                    matches.append((k, py_val, model_val))
        else:
            matched_endpoint = False
            if len(pytorch_outputs_np) == 1:
                end_pts_list = (
                    [end_points] if not isinstance(end_points, list) else end_points
                )
                for ep in end_pts_list:
                    if isinstance(ep, str) and ep in model_outputs:
                        matches.append((ep, pytorch_outputs_np[0], model_outputs[ep]))
                        matched_endpoint = True
                        break

            if not matched_endpoint:
                model_names = list(model_outputs.keys())
                for i, py_val in enumerate(pytorch_outputs_np):
                    if i < len(model_outs_list):
                        name = model_names[i] if i < len(model_names) else f"output_{i}"
                        matches.append((name, py_val, model_outs_list[i]))

        report = {}
        for name, py_val, raw_model_val in matches:
            curr_model_val = raw_model_val
            if py_val.shape != curr_model_val.shape:
                with contextlib.suppress(ValueError):
                    curr_model_val = curr_model_val.reshape(py_val.shape)

            mae = np.mean(np.abs(py_val - curr_model_val))
            mse = np.mean((py_val - curr_model_val) ** 2)

            py_flat = py_val.flatten()
            model_flat = curr_model_val.flatten()
            norm_py = np.linalg.norm(py_flat)
            norm_model = np.linalg.norm(model_flat)
            if norm_py > 0 and norm_model > 0:
                cos_sim = np.dot(py_flat, model_flat) / (norm_py * norm_model)
            else:
                cos_sim = 1.0 if norm_py == norm_model else 0.0

            all_close = np.allclose(py_val, curr_model_val, rtol=rtol, atol=atol)

            report[name] = {
                "mae": float(mae),
                "mse": float(mse),
                "cosine_similarity": float(cos_sim),
                "all_close": bool(all_close),
                "shape_match": py_val.shape == curr_model_val.shape,
            }

        return report

    def verify_against_pytorch(
        self,
        start_points: list[str | object] | str | object,
        end_points: list[str | object] | str | object,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> Any:
        """Decorator to verify PyTorch layer/function outputs against the model's sliced subgraph."""

        def decorator(func: Any) -> Any:
            from functools import wraps

            import numpy as np
            import torch

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                py_out = func(*args, **kwargs)
                inputs_dict = {}
                func_args = args
                if (
                    len(args) > 0
                    and hasattr(args[0], "__class__")
                    and isinstance(args[0], torch.nn.Module)
                ):
                    func_args = args[1:]

                for idx, arg in enumerate(func_args):
                    if isinstance(arg, torch.nn.Module):
                        continue
                    if isinstance(arg, torch.Tensor):
                        inputs_dict[f"input_{idx}"] = arg.cpu().detach().numpy()
                    elif isinstance(arg, np.ndarray):
                        inputs_dict[f"input_{idx}"] = arg

                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        inputs_dict[k] = v.cpu().detach().numpy()
                    elif isinstance(v, np.ndarray):
                        inputs_dict[k] = v

                try:
                    report = self.compare_outputs(
                        start_points=start_points,
                        end_points=end_points,
                        pytorch_module=func,
                        inputs_dict=inputs_dict,
                        rtol=rtol,
                        atol=atol,
                    )
                    wrapper.similarity_report = report
                except Exception as e:
                    log.warning("PyTorch verification comparison failed: %s", e)
                    wrapper.similarity_report = {"error": str(e)}

                return py_out

            return wrapper

        return decorator

    @property
    def passes(self) -> GraphPasses:
        """Return a namespace for common graph optimization passes."""
        return GraphPasses(self)


class GraphPasses:
    """Namespace for common graph optimization passes."""

    def __init__(self, graph: NeuronGraph) -> None:
        self.graph = graph

    def fuse_conv_bn(self) -> NeuronRewriter:
        """Fuse a Conv followed by BatchNormalization into a single Conv.

        The BatchNormalization scale/shift are folded into the Conv weights
        (and bias), producing an equivalent graph containing only the Conv
        operator. Returns a rewriter; call ``.build()`` to apply it.
        """
        import numpy as np

        from neuron_toolkit.pattern import Pattern

        conv = Pattern.op("Conv").capture("conv")
        bn = Pattern.op("BatchNormalization", conv).capture("bn")
        matches = self.graph.findall(bn)
        rewriter = self.graph.rewriter()
        if not matches:
            return rewriter

        backend = self.graph._backend
        tensor_map = self.graph.tensor_map

        def _const(name: str) -> np.ndarray | None:
            val = tensor_map.get(name)
            return val if isinstance(val, np.ndarray) else None

        for m in matches:
            conv_node = m.bindings["conv"]
            bn_node = m.bindings["bn"]
            bn_inputs = list(getattr(bn_node, "input", []))
            conv_inputs = list(getattr(conv_node, "input", []))
            if len(bn_inputs) != 5 or len(conv_inputs) < 2:
                continue

            scale = _const(bn_inputs[1])
            beta = _const(bn_inputs[2])
            mean = _const(bn_inputs[3])
            var = _const(bn_inputs[4])
            weight = _const(conv_inputs[1])
            if (
                scale is None
                or beta is None
                or mean is None
                or var is None
                or weight is None
            ):
                continue
            scale = np.asarray(scale)
            beta = np.asarray(beta)
            mean = np.asarray(mean)
            var = np.asarray(var)
            weight = np.asarray(weight)
            if not (scale.shape == beta.shape == mean.shape == var.shape):
                continue
            if weight.shape[0] != scale.shape[0]:
                continue

            bn_attrs = backend.get_node_attrs(bn_node)
            if bn_attrs.get("training_mode", 0) not in (0, False):
                continue
            eps = float(bn_attrs.get("epsilon", 1e-5))

            dtype = weight.dtype
            multiplier = (scale / np.sqrt(var + eps)).astype(dtype)
            scale_b = multiplier.reshape((scale.shape[0],) + (1,) * (weight.ndim - 1))
            new_weight = (weight * scale_b).astype(dtype)

            gamma_scale = multiplier
            if len(conv_inputs) > 2 and conv_inputs[2]:
                conv_bias = _const(conv_inputs[2])
                if conv_bias is None:
                    continue
                new_bias = (
                    np.asarray(conv_bias) * gamma_scale + beta - mean * gamma_scale
                ).astype(dtype)
            else:
                new_bias = (beta - mean * gamma_scale).astype(dtype)

            bn_name = getattr(bn_node, "name", None)
            fused_w = f"{bn_name or 'fused_conv_bn'}_weight"
            fused_b = f"{bn_name or 'fused_conv_bn'}_bias"
            rewriter.register_initializer(fused_w, new_weight)
            rewriter.register_initializer(fused_b, new_bias)

            rewriter.replace(
                [conv_node, bn_node],
                "Conv",
                inputs=[conv_inputs[0], fused_w, fused_b],
                outputs=list(getattr(bn_node, "output", [])),
                name=bn_name or "conv_bn_fused",
                **backend.get_node_attrs(conv_node),
            )
        return rewriter
