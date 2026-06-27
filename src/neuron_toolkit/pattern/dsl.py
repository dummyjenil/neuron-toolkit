from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuron_toolkit.graph import NeuronGraph
    from neuron_toolkit.pattern.models import MatchResult
    from neuron_toolkit.query import NeuronQuery

# Sentinel op strings
_WILDCARD = "__any__"
_ANY_OF = "__any_of__"
_CONST_PAT = "__const__"

MIN_ALTERNATIVES = 2


from neuron_toolkit.pattern._activations import ActivationMixin  # noqa: E402


class Pattern(ActivationMixin):
    """DSL for describing subgraph structures.

    A Pattern represents a symbolic subgraph that can be matched against an
    actual model. It supports wildcards, constant matching, attribute
    constraints, and output shape/dtype requirements.

    Example:
        >>> from neuron_toolkit.pattern.dsl import Pattern
        >>> p = Pattern.op("Add", Pattern.any(), Pattern.const(1.0))
        >>> p = p.capture("my_add").where(axis=1)
    """

    def __init__(
        self,
        op_type: str | None = None,
        inputs: list[Pattern] | None = None,
        value: object = None,
        *,
        _alternatives: list[Pattern] | None = None,
        _capture: str | None = None,
        _constraints: dict[str, object] | None = None,
        _rank: int | None = None,
        _dtype: str | None = None,
    ) -> None:
        """Initialize a Pattern node."""
        self.op_type = op_type
        self.inputs = inputs or []
        self.value = value
        self._alternatives = _alternatives or []
        self._capture = _capture
        self._constraints = _constraints or {}
        self._rank = _rank
        self._dtype = _dtype

    @property
    def alternatives(self) -> list[Pattern]:
        """Return the alternatives patterns list."""
        return self._alternatives

    @property
    def capture_name(self) -> str | None:
        """Return the capture name of the pattern."""
        return self._capture

    @property
    def constraints(self) -> dict[str, object]:
        """Return the constraints dict."""
        return self._constraints

    @property
    def rank(self) -> int | None:
        """Return the rank constraint."""
        return self._rank

    @property
    def dtype_str(self) -> str | None:
        """Return the dtype constraint."""
        return self._dtype

    @classmethod
    def any(cls) -> Pattern:
        """Create a wildcard pattern that matches any node."""
        return cls(op_type=_WILDCARD)

    @classmethod
    def const(cls, value: object) -> Pattern:
        """Create a pattern that matches a constant with specific value."""
        return cls(op_type=_CONST_PAT, value=value)

    @classmethod
    def op(cls, op_type: str, *input_patterns: Pattern | object) -> Pattern:
        """Create a pattern for a specific operator and its inputs.

        Args:
            op_type: The operator type.
            *input_patterns: Input patterns or raw values.
        """
        return cls(op_type=op_type, inputs=[_coerce(p) for p in input_patterns])

    @classmethod
    def any_of(cls, *alternatives: Pattern) -> Pattern:
        """Create a pattern that matches any of the given alternatives."""
        if len(alternatives) < MIN_ALTERNATIVES:
            msg = "any_of() requires at least 2 alternatives"
            raise ValueError(msg)
        return cls(op_type=_ANY_OF, _alternatives=list(alternatives))

    def capture(self, name: str) -> Pattern:
        """Assign a capture name to this pattern node."""
        return Pattern(
            self.op_type,
            list(self.inputs),
            self.value,
            _alternatives=list(self._alternatives),
            _capture=name,
            _constraints=dict(self._constraints),
            _rank=self._rank,
            _dtype=self._dtype,
        )

    def where(self, **constraints: object) -> Pattern:
        """Add attribute constraints to this pattern node."""
        merged = {**self._constraints, **constraints}
        return Pattern(
            self.op_type,
            list(self.inputs),
            self.value,
            _alternatives=list(self._alternatives),
            _capture=self._capture,
            _constraints=merged,
            _rank=self._rank,
            _dtype=self._dtype,
        )

    def with_output_rank(self, rank: int) -> Pattern:
        """Constrain the output tensor to have a specific rank."""
        return Pattern(
            self.op_type,
            list(self.inputs),
            self.value,
            _alternatives=list(self._alternatives),
            _capture=self._capture,
            _constraints=dict(self._constraints),
            _rank=rank,
            _dtype=self._dtype,
        )

    def with_dtype(self, dtype: str) -> Pattern:
        """Constrain the output tensor to have a specific data type."""
        return Pattern(
            self.op_type,
            list(self.inputs),
            self.value,
            _alternatives=list(self._alternatives),
            _capture=self._capture,
            _constraints=dict(self._constraints),
            _rank=self._rank,
            _dtype=dtype,
        )

    # --- Fusion Engine Integration ---

    def find(self, target: NeuronGraph | NeuronQuery) -> MatchResult | None:
        """Find the first match of this pattern in the target graph."""
        from neuron_toolkit.graph import NeuronGraph  # noqa: PLC0415

        if isinstance(target, NeuronGraph):
            return target.match(self)
        results = target.match_results(self)
        return results[0] if results else None

    def findall(self, target: NeuronGraph | NeuronQuery) -> list[MatchResult]:
        """Find all matches of this pattern in the target."""
        from neuron_toolkit.graph import NeuronGraph  # noqa: PLC0415

        if isinstance(target, NeuronGraph):
            return target.findall(self)
        return target.match_results(self)

    def filter(self, target: NeuronGraph | NeuronQuery) -> NeuronQuery:
        """Filter the target graph/query to nodes that match this pattern."""
        from neuron_toolkit.graph import NeuronGraph  # noqa: PLC0415

        if isinstance(target, NeuronGraph):
            return target.query().matches(self)
        return target.matches(self)

    # Arithmetic DSL
    def __add__(self, other: object) -> Pattern:
        """Return a pattern for Addition."""
        return Pattern("Add", [self, _coerce(other)])

    def __radd__(self, other: object) -> Pattern:
        """Return a pattern for Addition (right-hand side)."""
        return Pattern("Add", [_coerce(other), self])

    def __mul__(self, other: object) -> Pattern:
        """Return a pattern for Multiplication."""
        return Pattern("Mul", [self, _coerce(other)])

    def __rmul__(self, other: object) -> Pattern:
        """Return a pattern for Multiplication (right-hand side)."""
        return Pattern("Mul", [_coerce(other), self])

    def __pow__(self, power: object) -> Pattern:
        """Return a pattern for Power."""
        return Pattern("Pow", [self, _coerce(power)])

    def __sub__(self, other: object) -> Pattern:
        """Return a pattern for Subtraction."""
        return Pattern("Sub", [self, _coerce(other)])

    def __rsub__(self, other: object) -> Pattern:
        """Return a pattern for Subtraction (right-hand side)."""
        return Pattern("Sub", [_coerce(other), self])

    def __truediv__(self, other: object) -> Pattern:
        """Return a pattern for Division."""
        return Pattern("Div", [self, _coerce(other)])

    def __rtruediv__(self, other: object) -> Pattern:
        """Return a pattern for Division (right-hand side)."""
        return Pattern("Div", [_coerce(other), self])

    def __neg__(self) -> Pattern:
        """Return a pattern for Negation."""
        return Pattern("Neg", [self])

    def then(self, op_type: str | Pattern) -> Pattern:
        """Create a new pattern that consumes this pattern as input."""
        if isinstance(op_type, str):
            return Pattern(op_type, [self])
        # If it's a pattern, it must be an 'unbound' pattern (no inputs)
        if op_type.inputs:
            msg = (
                f"Cannot call .then() with a pattern that already has inputs: "
                f"{op_type.op_type}"
            )
            raise ValueError(msg)
        return Pattern(
            op_type.op_type,
            [self],
            op_type.value,
            _alternatives=list(op_type.alternatives),
            _capture=op_type.capture_name,
            _constraints=dict(op_type.constraints),
            _rank=op_type.rank,
            _dtype=op_type.dtype_str,
        )

    # --- Traversal API ---

    def nodes(self) -> list[Pattern]:
        """Return all unique pattern nodes in this pattern tree."""
        seen = set()
        result = []

        def _visit(p: Pattern) -> None:
            if id(p) not in seen:
                seen.add(id(p))
                result.append(p)
                for inp in p.inputs:
                    _visit(inp)
                for alt in p._alternatives:
                    _visit(alt)

        _visit(self)
        return result

    def walk(self) -> list[Pattern]:
        """Alias for nodes()."""
        return self.nodes()

    @property
    def parents(self) -> list[Pattern]:
        """Return the inputs of this pattern node."""
        return list(self.inputs)

    @property
    def children(self) -> list[Pattern]:
        """Patterns are defined backwards, so children is empty."""
        return []

    def __call__(self, *args: object) -> Pattern:
        """Use the pattern as a callable to specify its inputs."""
        if self.inputs:
            msg = f"Pattern '{self.op_type}' already has inputs"
            raise ValueError(msg)
        return Pattern(
            self.op_type,
            [_coerce(x) for x in args],
            self.value,
            _alternatives=list(self._alternatives),
            _capture=self._capture,
            _constraints=dict(self._constraints),
            _rank=self._rank,
            _dtype=self._dtype,
        )


def _coerce(x: object) -> Pattern:
    """Coerce a value into a constant Pattern."""
    return x if isinstance(x, Pattern) else Pattern(op_type=_CONST_PAT, value=x)
