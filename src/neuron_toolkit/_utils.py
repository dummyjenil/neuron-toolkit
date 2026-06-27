"""neuron_toolkit._utils — internal helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

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
        nodes: Sequence[object],
        tensor_map: Mapping[str, object],
        shape_info: ShapeInfo | None = None,
        backend: object | None = None,
    ) -> None:
        self.nodes = nodes
        self.tensor_map = tensor_map
        self.shape_info: ShapeInfo = shape_info or {}
        self.backend = backend
