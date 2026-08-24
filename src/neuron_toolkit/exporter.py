"""neuron_toolkit.exporter.

Export model graphs (ONNX, TFLite, unified NeuronGraph) to clean, structured
JSON format suitable for Netron-like visualizers and graph inspection tools.
Strictly excludes raw weight byte buffers, replacing them with statistical
analyses (sum, mean, min, max, std, sparsity, shape, dtype).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

log = logging.getLogger("neuron_toolkit.exporter")


def _sanitize_scalar(val: Any) -> Any:  # noqa: PLR0911
    """Convert scalar values (numpy, float, bytes) to JSON-friendly primitives."""
    if isinstance(val, (float, np.floating)):
        fval = float(val)
        if np.isnan(fval):
            return "NaN"
        if np.isinf(fval):
            return "Infinity" if fval > 0 else "-Infinity"
        return fval
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except UnicodeDecodeError:
            return f"<bytes len={len(val)}>"
    return val


def _sanitize_for_json(val: Any) -> Any:  # noqa: PLR0911
    """Recursively convert numpy types, bytes, and custom objects to standard JSON serializable primitives."""
    if val is None or isinstance(val, (bool, str, int)):
        return val
    if isinstance(val, (float, np.floating, np.integer, np.bool_, bytes)):
        return _sanitize_scalar(val)
    if isinstance(val, np.ndarray):
        if val.size <= 32:
            return [_sanitize_for_json(x) for x in val.tolist()]
        return {
            "type": "ndarray",
            "shape": list(val.shape),
            "dtype": str(val.dtype),
            "numel": int(val.size),
        }
    if isinstance(val, (list, tuple, set)):
        return [_sanitize_for_json(x) for x in val]
    if isinstance(val, Mapping):
        return {str(k): _sanitize_for_json(v) for k, v in val.items()}
    return str(val)


def compute_weight_stats(tensor: Any) -> dict[str, Any]:
    """Compute statistical analysis on a weight tensor without saving raw values.

    Calculates sum, mean, min, max, standard deviation, zero fraction,
    and checks for NaN/Inf.
    """
    if tensor is None:
        return {}

    arr = np.asarray(tensor)
    numel = int(arr.size)
    shape = list(arr.shape)
    dtype_str = str(arr.dtype)

    if numel == 0:
        return {
            "shape": shape,
            "dtype": dtype_str,
            "numel": 0,
            "stats": {
                "sum": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
                "zeros_count": 0,
                "zero_fraction": 0.0,
                "has_nan": False,
                "has_inf": False,
            },
        }

    # If numeric data type
    if np.issubdtype(arr.dtype, np.number):
        zeros_count = numel - int(np.count_nonzero(arr))
        zero_fraction = float(zeros_count / numel) if numel > 0 else 0.0

        if np.issubdtype(arr.dtype, np.integer):
            # Integers can never have NaN or Inf
            has_nan = False
            has_inf = False
            t_sum = float(np.sum(arr, dtype=np.float64))
            t_mean = float(np.mean(arr, dtype=np.float64))
            t_min = float(arr.min())
            t_max = float(arr.max())
            t_std = float(arr.std(dtype=np.float64))
        else:
            # Floating-point path: fast reduction without array copy
            with np.errstate(all="ignore"):
                t_sum = float(np.sum(arr, dtype=np.float64))
                t_min = float(arr.min())
                t_max = float(arr.max())

                # Fast check for NaN/Inf: if all scalar stats are finite, no NaN/Inf exist
                if np.isfinite(t_sum) and np.isfinite(t_min) and np.isfinite(t_max):
                    has_nan = False
                    has_inf = False
                    t_mean = float(np.mean(arr, dtype=np.float64))
                    t_std = float(arr.std(dtype=np.float64))
                else:
                    has_nan = bool(np.isnan(arr).any())
                    has_inf = bool(np.isinf(arr).any())
                    # Mask out non-finite values if possible for partial stats
                    finite_mask = np.isfinite(arr)
                    if finite_mask.any():
                        finite_arr = arr[finite_mask]
                        t_sum = float(np.sum(finite_arr, dtype=np.float64))
                        t_mean = float(np.mean(finite_arr, dtype=np.float64))
                        t_min = float(finite_arr.min())
                        t_max = float(finite_arr.max())
                        t_std = float(finite_arr.std(dtype=np.float64))
                    else:
                        t_sum = float("nan")
                        t_mean = float("nan")
                        t_min = float("nan")
                        t_max = float("nan")
                        t_std = float("nan")

        stats = {
            "sum": _sanitize_for_json(t_sum),
            "mean": _sanitize_for_json(t_mean),
            "min": _sanitize_for_json(t_min),
            "max": _sanitize_for_json(t_max),
            "std": _sanitize_for_json(t_std),
            "zeros_count": zeros_count,
            "zero_fraction": round(zero_fraction, 6),
            "has_nan": has_nan,
            "has_inf": has_inf,
        }
    else:
        # Non-numeric weights (e.g. strings or booleans)
        stats = {
            "sum": None,
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
            "zeros_count": int(np.count_nonzero(arr == 0))
            if arr.dtype == bool
            else 0,
            "zero_fraction": 0.0,
            "has_nan": False,
            "has_inf": False,
        }

    return {
        "shape": shape,
        "dtype": dtype_str,
        "numel": numel,
        "stats": stats,
    }


def _extract_onnx_full_shapes(model: Any) -> dict[str, dict[str, Any]]:
    """Extract detailed shape dimension lists and dtypes from an ONNX ModelProto."""
    from neuron_toolkit.backends.onnx.utils import _ONNX_DTYPE_TO_NP

    shapes: dict[str, dict[str, Any]] = {}

    def _process_vi(vi: Any) -> None:
        if not hasattr(vi, "type") or not vi.type.HasField("tensor_type"):
            return
        t = vi.type.tensor_type
        dtype = _ONNX_DTYPE_TO_NP.get(t.elem_type, "unknown")
        dim_list: list[Any] = []
        if t.HasField("shape"):
            for d in t.shape.dim:
                if d.HasField("dim_value"):
                    dim_list.append(d.dim_value)
                elif d.HasField("dim_param"):
                    dim_list.append(d.dim_param)
                else:
                    dim_list.append(None)
        shapes[vi.name] = {
            "name": vi.name,
            "shape": dim_list,
            "rank": len(dim_list),
            "dtype": dtype,
        }

    for vi in getattr(model.graph, "input", []):
        _process_vi(vi)
    for vi in getattr(model.graph, "value_info", []):
        _process_vi(vi)
    for vi in getattr(model.graph, "output", []):
        _process_vi(vi)

    return shapes


def _extract_tflite_full_shapes(
    subgraph: Any,
    quant_info: dict[str, Any] | None = None,
    sparse_info: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract detailed shape and dtype info from a TFLite subgraph."""
    from neuron_toolkit.backends.tflite.utils import _TFLITE_DTYPE_TO_NP

    shapes: dict[str, dict[str, Any]] = {}
    for i in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(i)
        name = tensor.Name().decode("utf-8")
        shape = [tensor.Shape(j) for j in range(tensor.ShapeLength())]
        dtype_code = tensor.Type()
        dtype = _TFLITE_DTYPE_TO_NP.get(dtype_code, "unknown")

        shape_entry: dict[str, Any] = {
            "name": name,
            "shape": shape,
            "rank": len(shape),
            "dtype": dtype,
        }

        # Check shape signature for dynamic dimensions
        if tensor.ShapeSignatureLength() > 0:
            shape_sig = [
                tensor.ShapeSignature(j) for j in range(tensor.ShapeSignatureLength())
            ]
            shape_entry["shape_signature"] = shape_sig

        if quant_info and name in quant_info:
            shape_entry["quantization"] = _sanitize_for_json(quant_info[name])
        if sparse_info and name in sparse_info:
            shape_entry["sparsity"] = _sanitize_for_json(sparse_info[name])

        shapes[name] = shape_entry

    return shapes


def export_graph_dict(source: Any) -> dict[str, Any]:
    """Export model or graph to a complete dictionary graph representation without raw weights."""
    # Unwrap NeuronGraph
    backend = getattr(source, "_backend", source)

    model_type = "Unknown"
    metadata: dict[str, Any] = {}
    nodes_raw: Sequence[Any] = getattr(backend, "nodes", [])
    tensor_map: Mapping[str, Any] = getattr(backend, "tensor_map", {})
    graph_inputs_set: set[str] = set(getattr(backend, "graph_inputs", set()))
    graph_outputs_set: set[str] = set(getattr(backend, "graph_outputs", set()))
    shape_info_basic: Mapping[str, Any] = getattr(backend, "shape_info", {})

    full_shapes: dict[str, dict[str, Any]] = {}

    # 1. Detect backend type & extract metadata
    if hasattr(backend, "model") and hasattr(backend.model, "graph"):
        # ONNX Parser
        model_type = "ONNX"
        model = backend.model
        metadata["producer_name"] = getattr(model, "producer_name", "")
        metadata["producer_version"] = getattr(model, "producer_version", "")
        metadata["domain"] = getattr(model, "domain", "")
        metadata["model_version"] = getattr(model, "model_version", 0)
        metadata["doc_string"] = getattr(model, "doc_string", "")
        metadata["ir_version"] = getattr(model, "ir_version", None)
        metadata["graph_name"] = getattr(model.graph, "name", "main_graph")
        metadata["graph_doc_string"] = getattr(model.graph, "doc_string", "")
        if hasattr(model, "opset_import"):
            metadata["opset_import"] = [
                {"domain": op.domain, "version": op.version}
                for op in model.opset_import
            ]
        try:
            full_shapes = _extract_onnx_full_shapes(model)
        except Exception as exc:
            log.debug("Could not extract full ONNX shapes: %s", exc)

    elif hasattr(backend, "_subgraph") and hasattr(backend, "_model"):
        # TFLite Parser
        model_type = "TFLite"
        subgraph = backend._subgraph
        model = backend._model
        metadata["version"] = getattr(model, "Version", lambda: 3)()
        metadata["description"] = (
            model.Description().decode("utf-8")
            if hasattr(model, "Description") and model.Description()
            else ""
        )
        quant_info = getattr(backend, "quantization_info", None)
        sparse_info = getattr(backend, "sparsity_info", None)
        try:
            full_shapes = _extract_tflite_full_shapes(
                subgraph, quant_info, sparse_info
            )
        except Exception as exc:
            log.debug("Could not extract full TFLite shapes: %s", exc)

    # Enrich full_shapes with tensor_map shapes and basic shape_info
    for name, (rank, dtype) in shape_info_basic.items():
        if name not in full_shapes:
            full_shapes[name] = {"name": name, "shape": None, "rank": rank, "dtype": dtype}

    # 2. Extract and analyze all weight initializers
    weights_summary_list: list[dict[str, Any]] = []
    weights_by_name: dict[str, dict[str, Any]] = {}
    total_elements = 0

    for name in sorted(tensor_map.keys()):
        arr = tensor_map.get(name)
        if arr is not None:
            stats_dict = compute_weight_stats(arr)
            stats_dict["name"] = name
            if name in full_shapes and "quantization" in full_shapes[name]:
                stats_dict["quantization"] = full_shapes[name]["quantization"]
            weights_by_name[name] = stats_dict
            weights_summary_list.append(stats_dict)
            total_elements += stats_dict["numel"]

            # Update shape info if missing
            if name not in full_shapes:
                full_shapes[name] = {
                    "name": name,
                    "shape": stats_dict["shape"],
                    "rank": len(stats_dict["shape"]),
                    "dtype": stats_dict["dtype"],
                }

    # 3. Extract Graph Inputs & Outputs
    inputs_list: list[dict[str, Any]] = []
    for inp_name in sorted(graph_inputs_set):
        info = full_shapes.get(inp_name, {})
        inputs_list.append(
            {
                "name": inp_name,
                "shape": info.get("shape"),
                "rank": info.get("rank"),
                "dtype": info.get("dtype"),
            }
        )

    outputs_list: list[dict[str, Any]] = []
    for out_name in sorted(graph_outputs_set):
        info = full_shapes.get(out_name, {})
        outputs_list.append(
            {
                "name": out_name,
                "shape": info.get("shape"),
                "rank": info.get("rank"),
                "dtype": info.get("dtype"),
            }
        )

    # 4. Extract Nodes
    nodes_list: list[dict[str, Any]] = []
    node_output_producer: dict[str, tuple[str, int]] = {}  # tensor_name -> (node_id, output_idx)

    for idx, node in enumerate(nodes_raw):
        op_type = getattr(node, "op_type", "Unknown")
        name = getattr(node, "name", "") or f"{op_type}_{idx}"
        node_id = f"node_{idx}_{name}" if name else f"node_{idx}"
        domain = getattr(node, "domain", "")

        # Extract attributes
        attrs_raw: dict[str, Any] = {}
        if hasattr(backend, "get_node_attrs"):
            try:
                attrs_raw = backend.get_node_attrs(node)
            except Exception:
                attrs_raw = getattr(node, "attrs", {})
        else:
            attrs_raw = getattr(node, "attrs", {})

        # Filter out large weight arrays from attributes if any
        sanitized_attrs = _sanitize_for_json(attrs_raw)

        # Inputs and Outputs
        node_inputs: list[str] = [
            str(inp) for inp in getattr(node, "input", []) if inp
        ]
        node_outputs: list[str] = [
            str(out) for out in getattr(node, "output", []) if out
        ]

        # Record output producer mapping
        for out_idx, out_name in enumerate(node_outputs):
            node_output_producer[out_name] = (node_id, out_idx)

        # Input & output shapes
        node_input_shapes: dict[str, Any] = {}
        for inp_name in node_inputs:
            if inp_name in full_shapes:
                node_input_shapes[inp_name] = full_shapes[inp_name]

        node_output_shapes: dict[str, Any] = {}
        for out_name in node_outputs:
            if out_name in full_shapes:
                node_output_shapes[out_name] = full_shapes[out_name]

        # Node weights stats
        node_weights: list[dict[str, Any]] = [
            weights_by_name[inp_name]
            for inp_name in node_inputs
            if inp_name in weights_by_name
        ]

        # Quantization / Sparsity
        node_quantization = None
        for out_name in node_outputs:
            if out_name in full_shapes and "quantization" in full_shapes[out_name]:
                node_quantization = full_shapes[out_name]["quantization"]
                break

        if (
            node_quantization is None
            and op_type in ("QuantizeLinear", "DequantizeLinear")
            and len(node_inputs) >= 2
        ):
            scale_name = node_inputs[1]
            scale_val = tensor_map.get(scale_name)
            zp_name = node_inputs[2] if len(node_inputs) >= 3 else None
            zp_val = tensor_map.get(zp_name) if zp_name else None
            node_quantization = {
                "scale": _sanitize_for_json(scale_val) if scale_val is not None else scale_name,
                "zero_point": _sanitize_for_json(zp_val) if zp_val is not None else zp_name,
                "axis": sanitized_attrs.get("axis", 1),
            }

        node_sparsity = None
        for inp_name in node_inputs:
            if inp_name in full_shapes and "sparsity" in full_shapes[inp_name]:
                node_sparsity = full_shapes[inp_name]["sparsity"]
                break

        node_entry = {
            "id": node_id,
            "name": name,
            "op_type": op_type,
            "domain": domain,
            "inputs": node_inputs,
            "outputs": node_outputs,
            "attributes": sanitized_attrs,
            "input_shapes": node_input_shapes,
            "output_shapes": node_output_shapes,
            "weights": node_weights,
            "quantization": node_quantization,
            "sparsity": node_sparsity,
        }
        nodes_list.append(node_entry)

    # 5. Extract Connections / Edges
    connections_list: list[dict[str, Any]] = []
    edge_idx = 0

    for target_node in nodes_list:
        target_id = target_node["id"]
        for in_idx, tensor_name in enumerate(target_node["inputs"]):
            is_weight = tensor_name in tensor_map
            is_graph_input = tensor_name in graph_inputs_set

            source_id: str
            source_out_idx: int = 0
            if tensor_name in node_output_producer:
                source_id, source_out_idx = node_output_producer[tensor_name]
            elif is_graph_input:
                source_id = f"__input__:{tensor_name}"
            elif is_weight:
                source_id = f"__weight__:{tensor_name}"
            else:
                source_id = f"__constant__:{tensor_name}"

            tensor_info = full_shapes.get(tensor_name, {})

            connection_entry = {
                "id": f"edge_{edge_idx}",
                "source": source_id,
                "target": target_id,
                "tensor_name": tensor_name,
                "shape": tensor_info.get("shape"),
                "rank": tensor_info.get("rank"),
                "dtype": tensor_info.get("dtype"),
                "source_output_index": source_out_idx,
                "target_input_index": in_idx,
                "is_weight": is_weight,
            }
            connections_list.append(connection_entry)
            edge_idx += 1

    # Connect node outputs to graph outputs if present
    for target_out in outputs_list:
        out_name = target_out["name"]
        if out_name in node_output_producer:
            source_id, source_out_idx = node_output_producer[out_name]
            tensor_info = full_shapes.get(out_name, {})
            connection_entry = {
                "id": f"edge_{edge_idx}",
                "source": source_id,
                "target": f"__output__:{out_name}",
                "tensor_name": out_name,
                "shape": tensor_info.get("shape"),
                "rank": tensor_info.get("rank"),
                "dtype": tensor_info.get("dtype"),
                "source_output_index": source_out_idx,
                "target_input_index": 0,
                "is_weight": False,
            }
            connections_list.append(connection_entry)
            edge_idx += 1

    # Update metadata counts
    metadata["num_nodes"] = len(nodes_list)
    metadata["num_edges"] = len(connections_list)
    metadata["num_inputs"] = len(inputs_list)
    metadata["num_outputs"] = len(outputs_list)
    metadata["num_weights"] = len(weights_summary_list)
    metadata["total_weight_params"] = total_elements

    return {
        "format": "neuron_toolkit_graph_v1",
        "model_type": model_type,
        "metadata": metadata,
        "inputs": inputs_list,
        "outputs": outputs_list,
        "nodes": nodes_list,
        "connections": connections_list,
        "edges": connections_list,  # Convenience alias
        "weights_summary": {
            "total_tensors": len(weights_summary_list),
            "total_elements": total_elements,
            "tensors": weights_summary_list,
        },
    }


def export_graph_json(
    source: Any,
    path: str | None = None,
    indent: int = 2,
) -> str | None:
    """Export model or graph to a JSON string or save to a file path.

    Parameters:
        source: A NeuronGraph, ONNXParser, TFLiteParser, or compatible graph object.
        path: Optional file path to write the JSON to.
        indent: JSON indentation spaces (default 2).

    Returns:
        JSON string if path is None, otherwise writes to path and returns None.
    """
    graph_dict = export_graph_dict(source)
    json_str = json.dumps(
        graph_dict,
        indent=indent,
        default=_sanitize_for_json,
    )

    if path is not None:
        from pathlib import Path

        Path(path).write_text(json_str, encoding="utf-8")
        return None

    return json_str


_NP_DTYPE_TO_SAFETENSORS: dict[str, str] = {
    "float32": "F32",
    "float64": "F64",
    "float16": "F16",
    "int8": "I8",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
    "uint8": "U8",
    "uint16": "U16",
    "uint32": "U32",
    "uint64": "U64",
    "bool": "BOOL",
}

_SAFETENSORS_TO_NP_DTYPE: dict[str, str] = {
    v: k for k, v in _NP_DTYPE_TO_SAFETENSORS.items()
}


def export_safetensors(
    source: Any,
    path: str,
    metadata: dict[str, str] | None = None,
) -> None:
    """Save all initializers and weights of the model into a standard .safetensors file.

    Parameters:
        source: A NeuronGraph, backend parser, or mapping of tensor names to NumPy arrays.
        path: Destination file path (e.g. 'model.safetensors').
        metadata: Optional string dictionary for safetensors __metadata__ header.
    """
    import struct
    from pathlib import Path

    # Extract tensor dictionary
    backend = getattr(source, "_backend", source)
    if hasattr(backend, "tensor_map"):
        tensor_dict = dict(backend.tensor_map)
    elif isinstance(source, Mapping):
        tensor_dict = dict(source)
    else:
        tensor_dict = {}

    # Built-in zero-dependency compliant Safetensors serializer
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    data_buffers: list[bytes] = []
    current_offset = 0

    for name in sorted(tensor_dict.keys()):
        arr = tensor_dict[name]
        if arr is None:
            continue
        np_arr = np.ascontiguousarray(arr)
        dtype_str = str(np_arr.dtype)
        st_dtype = _NP_DTYPE_TO_SAFETENSORS.get(dtype_str, "F32")
        raw_bytes = np_arr.tobytes()
        byte_len = len(raw_bytes)

        header[name] = {
            "dtype": st_dtype,
            "shape": list(np_arr.shape),
            "data_offsets": [current_offset, current_offset + byte_len],
        }
        data_buffers.append(raw_bytes)
        current_offset += byte_len

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_bytes)

    with Path(path).open("wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        f.writelines(data_buffers)


def load_safetensors(path: str) -> dict[str, np.ndarray]:
    """Load tensors from a .safetensors file into a dictionary of NumPy arrays."""
    import struct
    from pathlib import Path

    with Path(path).open("rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            msg = "Invalid safetensors file: insufficient header length."
            raise ValueError(msg)
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        header_json_bytes = f.read(header_len)
        header = json.loads(header_json_bytes.decode("utf-8"))

        data_start_offset = 8 + header_len
        result: dict[str, np.ndarray] = {}

        for key, info in header.items():
            if key == "__metadata__":
                continue
            st_dtype = info["dtype"]
            shape = info["shape"]
            start_off, end_off = info["data_offsets"]
            byte_len = end_off - start_off

            f.seek(data_start_offset + start_off)
            raw_bytes = f.read(byte_len)

            np_dtype = _SAFETENSORS_TO_NP_DTYPE.get(st_dtype, "float32")
            arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)
            result[key] = arr

        return result


_NP_DTYPE_TO_ONNX: dict[str, int] = {
    "float32": 1,   # TensorProto.FLOAT
    "uint8": 2,     # TensorProto.UINT8
    "int8": 3,      # TensorProto.INT8
    "uint16": 4,    # TensorProto.UINT16
    "int16": 5,     # TensorProto.INT16
    "int32": 6,     # TensorProto.INT32
    "int64": 7,     # TensorProto.INT64
    "string": 8,    # TensorProto.STRING
    "object": 8,    # TensorProto.STRING
    "bool": 9,      # TensorProto.BOOL
    "float16": 10,  # TensorProto.FLOAT16
    "float64": 11,  # TensorProto.DOUBLE
    "uint32": 12,   # TensorProto.UINT32
    "uint64": 13,   # TensorProto.UINT64
    "complex64": 14,  # TensorProto.COMPLEX64
    "complex128": 15,  # TensorProto.COMPLEX128
    "bfloat16": 16,  # TensorProto.BFLOAT16
}


def _random_tensor_for_dtype(
    shape: Sequence[int],
    dtype_str: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a pseudo-random or constant tensor matching the given shape and dtype."""
    if rng is None:
        rng = np.random.default_rng()

    clean_shape = tuple(
        int(dim) if isinstance(dim, (int, np.integer)) else 1 for dim in shape
    )

    if dtype_str in ("float32", "float64", "float16", "bfloat16"):
        raw = rng.standard_normal(clean_shape)
        return raw.astype(dtype_str if dtype_str != "bfloat16" else "float32")
    if dtype_str in ("int32", "int64", "int16", "int8"):
        return rng.integers(0, 10, size=clean_shape, dtype=dtype_str)
    if dtype_str in ("uint8", "uint16", "uint32", "uint64"):
        return rng.integers(0, 10, size=clean_shape, dtype=dtype_str)
    if dtype_str == "bool":
        return rng.choice([True, False], size=clean_shape)
    return np.zeros(clean_shape, dtype=dtype_str)


def build_onnx_model_from_json(
    json_dict_or_str: dict[str, Any] | str,
    weights: dict[str, np.ndarray] | str | None = None,
    seed: int | None = None,
) -> Any:
    """Rebuild a complete ONNX ModelProto from a JSON graph representation and optional weights.

    Parameters:
        json_dict_or_str: Python dictionary or JSON string/filepath from `to_json()` / `to_dict()`.
        weights: Optional dictionary of NumPy arrays, or path to a .safetensors file.
                 If None, weights and initializers will be randomly initialized.
        seed: Optional integer seed for reproducible random weight initialization.

    Returns:
        onnx.ModelProto instance.
    """
    from pathlib import Path

    import onnx
    from onnx import helper, numpy_helper

    if isinstance(json_dict_or_str, str):
        if json_dict_or_str.strip().startswith("{"):
            graph_dict = json.loads(json_dict_or_str)
        else:
            graph_dict = json.loads(Path(json_dict_or_str).read_text(encoding="utf-8"))
    else:
        graph_dict = json_dict_or_str

    weights_map: dict[str, np.ndarray] = {}
    if isinstance(weights, str):
        weights_map = load_safetensors(weights)
    elif isinstance(weights, Mapping):
        weights_map = dict(weights)

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    metadata = graph_dict.get("metadata", {})
    graph_name = metadata.get("graph_name", "reconstructed_graph")
    graph_doc = metadata.get("graph_doc_string", "")

    # 1. Rebuild Initializers / Weights
    initializers_dict: dict[str, np.ndarray] = {}

    weights_summary = graph_dict.get("weights_summary", {})
    tensors_list = weights_summary.get("tensors", [])
    for t_info in tensors_list:
        name = t_info.get("name")
        if not name:
            continue
        shape = t_info.get("shape", [])
        dtype_str = t_info.get("dtype", "float32")
        if name in weights_map:
            initializers_dict[name] = np.ascontiguousarray(weights_map[name])
        else:
            initializers_dict[name] = _random_tensor_for_dtype(shape, dtype_str, rng)

    # Include any extra weight passed in weights_map
    for name, arr in weights_map.items():
        if name not in initializers_dict:
            initializers_dict[name] = np.ascontiguousarray(arr)

    # 2. Build ONNX Initializer TensorProtos
    onnx_initializers = [
        numpy_helper.from_array(arr, name=name)
        for name, arr in sorted(initializers_dict.items())
    ]
    initializer_names = set(initializers_dict.keys())

    # 3. Build Graph Inputs
    graph_inputs = []
    for inp in graph_dict.get("inputs", []):
        name = inp["name"]
        dtype_str = inp.get("dtype", "float32")
        elem_type = _NP_DTYPE_TO_ONNX.get(dtype_str, onnx.TensorProto.FLOAT)
        shape = inp.get("shape")
        vi = helper.make_tensor_value_info(name, elem_type, shape)
        graph_inputs.append(vi)

    # 4. Build Graph Outputs
    graph_outputs = []
    for out in graph_dict.get("outputs", []):
        name = out["name"]
        dtype_str = out.get("dtype", "float32")
        elem_type = _NP_DTYPE_TO_ONNX.get(dtype_str, onnx.TensorProto.FLOAT)
        shape = out.get("shape")
        vi = helper.make_tensor_value_info(name, elem_type, shape)
        graph_outputs.append(vi)

    # 5. Build Graph Nodes
    graph_nodes = []
    for node_info in graph_dict.get("nodes", []):
        op_type = node_info.get("op_type", "Unknown")
        name = node_info.get("name", "")
        domain = node_info.get("domain", "") or ""
        inputs = node_info.get("inputs", [])
        outputs = node_info.get("outputs", [])
        raw_attrs = node_info.get("attributes", {})

        cleaned_attrs = {}
        for k, v in raw_attrs.items():
            if op_type == "Constant" and k == "value":
                if isinstance(v, (list, np.ndarray)):
                    cleaned_attrs[k] = numpy_helper.from_array(np.asarray(v), name="value")
                elif isinstance(v, dict) and v.get("type") == "ndarray":
                    shape = v.get("shape", [1])
                    dtype = v.get("dtype", "float32")
                    arr = _random_tensor_for_dtype(shape, dtype, rng)
                    cleaned_attrs[k] = numpy_helper.from_array(arr, name="value")
                elif isinstance(v, onnx.TensorProto):
                    cleaned_attrs[k] = v
                else:
                    cleaned_attrs[k] = v
            else:
                cleaned_attrs[k] = v

        node_kw = {}
        if domain:
            node_kw["domain"] = domain

        node_proto = helper.make_node(
            op_type,
            inputs=inputs,
            outputs=outputs,
            name=name,
            **node_kw,
            **cleaned_attrs,
        )
        graph_nodes.append(node_proto)

    # 6. Build Value Info (Intermediate Shapes if any)
    value_infos = []
    for node_info in graph_dict.get("nodes", []):
        output_shapes = node_info.get("output_shapes", {})
        for out_name, shape_meta in output_shapes.items():
            if any(out["name"] == out_name for out in graph_dict.get("outputs", [])):
                continue
            if out_name in initializer_names:
                continue
            shape = shape_meta.get("shape")
            dtype_str = shape_meta.get("dtype", "float32")
            elem_type = _NP_DTYPE_TO_ONNX.get(dtype_str, onnx.TensorProto.FLOAT)
            if shape is not None:
                vi = helper.make_tensor_value_info(out_name, elem_type, shape)
                value_infos.append(vi)

    # 7. Make Graph
    graph_proto = helper.make_graph(
        nodes=graph_nodes,
        name=graph_name,
        inputs=graph_inputs,
        outputs=graph_outputs,
        initializer=onnx_initializers,
        doc_string=graph_doc,
        value_info=value_infos,
    )

    # 8. Opset Imports
    opset_imports = [
        helper.make_opsetid(opset.get("domain", ""), opset.get("version", 17))
        for opset in metadata.get("opset_import", [])
    ]
    if not opset_imports:
        opset_imports = [helper.make_opsetid("", 17)]

    # 9. Make Model
    model_kwargs: dict[str, Any] = {
        "producer_name": metadata.get("producer_name", "neuron_toolkit"),
        "producer_version": metadata.get("producer_version", "0.1.0"),
        "doc_string": metadata.get("doc_string", ""),
        "domain": metadata.get("domain", ""),
        "model_version": metadata.get("model_version", 1),
        "opset_imports": opset_imports,
    }
    if metadata.get("ir_version") is not None:
        model_kwargs["ir_version"] = metadata["ir_version"]

    return helper.make_model(
        graph_proto,
        **model_kwargs,
    )


def load_graph_from_json(
    json_source: dict[str, Any] | str,
    weights_source: dict[str, np.ndarray] | str | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> Any:
    """Load and rebuild a complete NeuronGraph from a JSON graph representation and optional weights.

    Parameters:
        json_source: JSON file path, JSON string, or graph dictionary.
        weights_source: Optional path to a .safetensors file or dictionary of NumPy arrays.
                        If omitted, weights will be randomly initialized.
        seed: Optional random seed for weight initialization.
        kwargs: Additional arguments passed to NeuronGraph.

    Returns:
        NeuronGraph instance containing the fully reconstructed model graph.
    """
    from neuron_toolkit.graph import NeuronGraph

    model_proto = build_onnx_model_from_json(
        json_source,
        weights=weights_source,
        seed=seed,
    )
    return NeuronGraph(model_proto, **kwargs)

