"""neuron_toolkit._utils — internal helpers."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("neuron_toolkit")


# ShapeInfo: value_name → (rank | None, dtype_str | None)
ShapeInfo = dict[str, tuple[int | None, str | None]]


# ---------------------------------------------------------------------------
# Graph shim
# ---------------------------------------------------------------------------


class _GraphShim:
    """Lightweight stand-in for ModelProto used inside PatternDetector."""

    def __init__(
        self,
        nodes: list[Any],
        tensor_map: dict[str, Any],
        shape_info: ShapeInfo | None = None,
        backend: Any | None = None,
    ) -> None:
        self.nodes = nodes
        self.tensor_map = tensor_map
        self.shape_info: ShapeInfo = shape_info or {}
        self.backend = backend
