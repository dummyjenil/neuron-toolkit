"""Performance testing utilities for timing and memory tracking."""

from __future__ import annotations

import functools
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


class PerformanceTracker:
    """Context manager and helper to record wall-clock execution time and peak memory footprint."""

    def __init__(self, op_name: str, scale_info: str = ""):
        self.op_name = op_name
        self.scale_info = scale_info
        self.start_time: float = 0.0
        self.duration_sec: float = 0.0
        self.peak_memory_mb: float = 0.0
        self.current_memory_mb: float = 0.0

    def __enter__(self) -> PerformanceTracker:
        tracemalloc.start()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.duration_sec = time.perf_counter() - self.start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.current_memory_mb = current / (1024 * 1024)
        self.peak_memory_mb = peak / (1024 * 1024)
        self.log_summary()

    def log_summary(self) -> None:
        print(f"\n{'=' * 75}")
        print(f" PERF METRIC | Operation : {self.op_name}")
        if self.scale_info:
            print(f"             | Scale     : {self.scale_info}")
        print(f"             | Time      : {self.duration_sec * 1000:.2f} ms ({self.duration_sec:.4f} s)")
        print(f"             | Peak Mem  : {self.peak_memory_mb:.3f} MB")
        print(f"             | Net Alloc : {self.current_memory_mb:.3f} MB")
        print(f"{'=' * 75}\n")


def profile_perf(op_name: str, scale_info: str = ""):
    """Decorator to measure time and memory of a benchmark function."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with PerformanceTracker(op_name=op_name, scale_info=scale_info):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def create_large_onnx_model(num_blocks: int = 1000, in_channels: int = 16, spatial: int = 14) -> onnx.ModelProto:
    """Build a heavy ONNX modelProto in-memory containing repetitive deep layers.

    Each block consists of:
    Conv2D -> BatchNormalization -> Add (Residual) -> Relu
    """
    rng = np.random.default_rng(42)
    x_shape = [1, in_channels, spatial, spatial]
    inputs = [helper.make_tensor_value_info("X_in", TensorProto.FLOAT, x_shape)]
    outputs = []
    nodes = []
    initializers = []

    prev_tensor = "X_in"

    for i in range(num_blocks):
        w_name = f"w_{i}"
        conv_out = f"conv_out_{i}"
        bn_scale = f"bn_scale_{i}"
        bn_bias = f"bn_bias_{i}"
        bn_mean = f"bn_mean_{i}"
        bn_var = f"bn_var_{i}"
        bn_out = f"bn_out_{i}"
        add_out = f"add_out_{i}"
        relu_out = f"relu_out_{i}"

        # Weights
        w = rng.normal(size=(in_channels, in_channels, 3, 3)).astype(np.float32)
        scale = rng.uniform(0.5, 2.0, (in_channels,)).astype(np.float32)
        bias = rng.normal(size=(in_channels,)).astype(np.float32)
        mean = rng.normal(size=(in_channels,)).astype(np.float32)
        var = rng.uniform(0.5, 1.5, (in_channels,)).astype(np.float32)

        initializers.extend([
            numpy_helper.from_array(w, w_name),
            numpy_helper.from_array(scale, bn_scale),
            numpy_helper.from_array(bias, bn_bias),
            numpy_helper.from_array(mean, bn_mean),
            numpy_helper.from_array(var, bn_var),
        ])

        # 1. Conv
        nodes.append(helper.make_node(
            "Conv",
            inputs=[prev_tensor, w_name],
            outputs=[conv_out],
            name=f"conv_{i}",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ))

        # 2. BatchNormalization
        nodes.append(helper.make_node(
            "BatchNormalization",
            inputs=[conv_out, bn_scale, bn_bias, bn_mean, bn_var],
            outputs=[bn_out],
            name=f"bn_{i}",
        ))

        # 3. Add (Residual connection with prev_tensor)
        nodes.append(helper.make_node(
            "Add",
            inputs=[bn_out, prev_tensor],
            outputs=[add_out],
            name=f"add_{i}",
        ))

        # 4. Relu
        nodes.append(helper.make_node(
            "Relu",
            inputs=[add_out],
            outputs=[relu_out],
            name=f"relu_{i}",
        ))

        prev_tensor = relu_out

    outputs.append(helper.make_tensor_value_info(prev_tensor, TensorProto.FLOAT, x_shape))

    graph = helper.make_graph(
        nodes,
        "heavy_onnx_benchmark",
        inputs,
        outputs,
        initializer=initializers,
    )
    return helper.make_model(graph, producer_name="neuron_toolkit_perf")


def build_tflite_model_internal(
    codes: list[int],
    tensors: list[dict[str, Any]],
    ops: list[dict[str, Any]],
    inputs: list[int],
    outputs: list[int],
    buffers: list[bytes | None],
    identifier: bytes = b"TFL3",
) -> bytes:
    """Build a TFLite flatbuffer model in-memory."""
    import flatbuffers
    import tflite

    builder = flatbuffers.Builder(2048)

    tensor_offsets = []
    for t in tensors:
        name = builder.CreateString(t["name"])
        tflite.TensorStartShapeVector(builder, len(t["shape"]))
        for s in reversed(t["shape"]):
            builder.PrependInt32(s)
        shape_vec = builder.EndVector()

        tflite.TensorStart(builder)
        tflite.TensorAddName(builder, name)
        tflite.TensorAddShape(builder, shape_vec)
        tflite.TensorAddType(builder, t["dtype"])
        tflite.TensorAddBuffer(builder, t["buffer"])
        tensor_offsets.append(tflite.TensorEnd(builder))

    tflite.SubGraphStartTensorsVector(builder, len(tensor_offsets))
    for off in reversed(tensor_offsets):
        builder.PrependUOffsetTRelative(off)
    tensors_vec = builder.EndVector()

    op_offsets = []
    for op in ops:
        tflite.OperatorStartInputsVector(builder, len(op["inputs"]))
        for x in reversed(op["inputs"]):
            builder.PrependInt32(x)
        inputs_vec = builder.EndVector()

        tflite.OperatorStartOutputsVector(builder, len(op["outputs"]))
        for x in reversed(op["outputs"]):
            builder.PrependInt32(x)
        outputs_vec = builder.EndVector()

        builtin_options_type = op.get("builtin_options_type", 0)
        builtin_options = op.get("builtin_options", 0)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, op["index"])
        tflite.OperatorAddInputs(builder, inputs_vec)
        tflite.OperatorAddOutputs(builder, outputs_vec)
        if builtin_options_type:
            tflite.OperatorAddBuiltinOptionsType(builder, builtin_options_type)
        if builtin_options:
            tflite.OperatorAddBuiltinOptions(builder, builtin_options)
        op_offsets.append(tflite.OperatorEnd(builder))

    tflite.SubGraphStartOperatorsVector(builder, len(op_offsets))
    for off in reversed(op_offsets):
        builder.PrependUOffsetTRelative(off)
    ops_vec = builder.EndVector()

    tflite.SubGraphStartInputsVector(builder, len(inputs))
    for i in reversed(inputs):
        builder.PrependInt32(i)
    inputs_vec = builder.EndVector()

    tflite.SubGraphStartOutputsVector(builder, len(outputs))
    for o in reversed(outputs):
        builder.PrependInt32(o)
    outputs_vec = builder.EndVector()

    tflite.SubGraphStart(builder)
    tflite.SubGraphAddTensors(builder, tensors_vec)
    tflite.SubGraphAddInputs(builder, inputs_vec)
    tflite.SubGraphAddOutputs(builder, outputs_vec)
    tflite.SubGraphAddOperators(builder, ops_vec)
    subgraph = tflite.SubGraphEnd(builder)

    tflite.ModelStartSubgraphsVector(builder, 1)
    builder.PrependUOffsetTRelative(subgraph)
    subgraphs_vec = builder.EndVector()

    opcode_offsets = []
    for code in codes:
        tflite.OperatorCodeStart(builder)
        tflite.OperatorCodeAddDeprecatedBuiltinCode(builder, min(code, 127))
        tflite.OperatorCodeAddBuiltinCode(builder, code)
        opcode_offsets.append(tflite.OperatorCodeEnd(builder))

    tflite.ModelStartOperatorCodesVector(builder, len(opcode_offsets))
    for off in reversed(opcode_offsets):
        builder.PrependUOffsetTRelative(off)
    opcodes_vec = builder.EndVector()

    buffer_offsets = []
    for data in buffers:
        data_vec = 0
        if data:
            data_vec = builder.CreateByteVector(data)
        tflite.BufferStart(builder)
        if data_vec:
            tflite.BufferAddData(builder, data_vec)
        buffer_offsets.append(tflite.BufferEnd(builder))

    tflite.ModelStartBuffersVector(builder, len(buffer_offsets))
    for off in reversed(buffer_offsets):
        builder.PrependUOffsetTRelative(off)
    buffers_vec = builder.EndVector()

    tflite.ModelStart(builder)
    tflite.ModelAddVersion(builder, 3)
    tflite.ModelAddSubgraphs(builder, subgraphs_vec)
    tflite.ModelAddOperatorCodes(builder, opcodes_vec)
    tflite.ModelAddBuffers(builder, buffers_vec)
    model = tflite.ModelEnd(builder)

    builder.Finish(model, file_identifier=identifier)
    return bytes(builder.Output())


def create_large_tflite_model_bytes(num_layers: int = 500) -> bytes:
    """Build a heavy TFLite model flatbuffer in-memory with hundreds of operators."""
    import tflite

    tensors = [
        {"name": "input_0", "shape": [1, 64, 64, 3], "dtype": tflite.TensorType.FLOAT32, "buffer": 0}
    ]
    ops = []

    # Opcodes: 0 -> ABS, 1 -> RELU, 2 -> SQRT
    codes = [
        tflite.BuiltinOperator.ABS,
        tflite.BuiltinOperator.RELU,
        tflite.BuiltinOperator.SQRT,
    ]

    for i in range(num_layers):
        out_name = f"tensor_{i+1}"
        tensors.append({"name": out_name, "shape": [1, 64, 64, 3], "dtype": tflite.TensorType.FLOAT32, "buffer": 0})
        op_code_idx = i % 3
        ops.append({"index": op_code_idx, "inputs": [i], "outputs": [i + 1]})

    buffers = [None]  # Buffer 0 must be empty

    return build_tflite_model_internal(
        codes=codes,
        tensors=tensors,
        ops=ops,
        inputs=[0],
        outputs=[num_layers],
        buffers=buffers,
    )
