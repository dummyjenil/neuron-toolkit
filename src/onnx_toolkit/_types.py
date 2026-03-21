"""
onnx_toolkit._types
====================
Shared type aliases used across the package.
"""

from typing import Dict
import numpy as np

# Weight tensors keyed by initializer name
TensorMap = Dict[str, np.ndarray]
