from __future__ import annotations

# Sentinel op strings
_WILDCARD = "__any__"
_ANY_OF = "__any_of__"
_CONST_PAT = "__const__"

MIN_ALTERNATIVES = 2


class Pattern:
    """DSL for describing ONNX subgraph structures."""

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

    @classmethod
    def any(cls) -> Pattern:
        """Create a wildcard pattern that matches any node."""
        return cls(op_type=_WILDCARD)

    @classmethod
    def const(cls, value: object) -> Pattern:
        """Create a pattern that matches a constant with specific value."""
        return cls(op_type=_CONST_PAT, value=value)

    @classmethod
    def op(cls, op_type: str, *input_patterns: Pattern) -> Pattern:
        """Create a pattern for a specific operator and its inputs."""
        return cls(op_type=op_type, inputs=list(input_patterns))

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

    def __call__(self, x: object) -> Pattern:
        """Use the pattern as a callable to specify its input."""
        if self.inputs:
            msg = f"Pattern '{self.op_type}' already has inputs"
            raise ValueError(msg)
        return Pattern(
            self.op_type,
            [_coerce(x)],
            self.value,
            _alternatives=list(self._alternatives),
            _capture=self._capture,
            _constraints=dict(self._constraints),
            _rank=self._rank,
            _dtype=self._dtype,
        )

    # Activation helpers
    @classmethod
    def relu(cls, x: Pattern) -> Pattern:
        """Create a Relu pattern."""
        return cls.op("Relu")(x)

    @classmethod
    def sigmoid(cls, x: Pattern) -> Pattern:
        """Create a Sigmoid pattern."""
        return cls.op("Sigmoid")(x)

    @classmethod
    def tanh(cls, x: Pattern) -> Pattern:
        """Create a Tanh pattern."""
        return cls.op("Tanh")(x)

    @classmethod
    def leaky_relu(cls, x: Pattern) -> Pattern:
        """Create a LeakyRelu pattern."""
        return cls.op("LeakyRelu")(x)

    @classmethod
    def elu(cls, x: Pattern) -> Pattern:
        """Create an Elu pattern."""
        return cls.op("Elu")(x)

    @classmethod
    def selu(cls, x: Pattern) -> Pattern:
        """Create a Selu pattern."""
        return cls.op("Selu")(x)

    @classmethod
    def softplus(cls, x: Pattern) -> Pattern:
        """Create a Softplus pattern."""
        return cls.op("Softplus")(x)

    @classmethod
    def softsign(cls, x: Pattern) -> Pattern:
        """Create a Softsign pattern."""
        return cls.op("Softsign")(x)

    @classmethod
    def hardsigmoid(cls, x: Pattern) -> Pattern:
        """Create a HardSigmoid pattern."""
        return cls.op("HardSigmoid")(x)

    @classmethod
    def hardswish(cls, x: Pattern) -> Pattern:
        """Create a HardSwish pattern."""
        return x * cls.hardsigmoid(x)

    @classmethod
    def silu(cls, x: Pattern) -> Pattern:
        """Create a SiLU pattern."""
        return x * cls.sigmoid(x)

    @classmethod
    def swish(cls, x: Pattern) -> Pattern:
        """Create a Swish pattern (alias for SiLU)."""
        return x * cls.sigmoid(x)

    @classmethod
    def gelu(cls, x: Pattern) -> Pattern:
        """Create a GeLU pattern (standard approximation)."""
        return x * (cls.op("Erf")(x / cls.const(1.41421356237)) + cls.const(1.0)) * cls.const(0.5)

    @classmethod
    def gelu_tanh(cls, x: Pattern) -> Pattern:
        """Create a GeLU pattern (Tanh approximation)."""
        return (
            cls.const(0.5)
            * x
            * (
                cls.const(1.0)
                + cls.tanh(
                    cls.const(0.7978845608) * (x + cls.const(0.044715) * (x ** cls.const(3.0)))
                )
            )
        )

    @classmethod
    def mish(cls, x: Pattern) -> Pattern:
        """Create a Mish activation pattern."""
        return x * cls.tanh(cls.softplus(x))

    @classmethod
    def relu6(cls, x: Pattern) -> Pattern:
        """Create a ReLU6 pattern."""
        return Pattern("Clip", [_coerce(x), cls.const(0.0), cls.const(6.0)])

    @classmethod
    def softmax(cls, x: Pattern) -> Pattern:
        """Create a Softmax pattern."""
        return cls.op("Softmax")(x)

    @classmethod
    def log_softmax(cls, x: Pattern) -> Pattern:
        """Create a LogSoftmax pattern."""
        return cls.op("LogSoftmax")(x)

    @classmethod
    def prelu(cls, x: Pattern, slope: Pattern) -> Pattern:
        """Create a PRelu pattern."""
        return cls.op("PRelu", x, slope)

    @classmethod
    def thresholded_relu(cls, x: Pattern) -> Pattern:
        """Create a ThresholdedRelu pattern."""
        return cls.op("ThresholdedRelu")(x)


def _coerce(x: object) -> Pattern:
    """Coerce a value (number, string, etc.) into a constant Pattern."""
    return x if isinstance(x, Pattern) else Pattern(op_type=_CONST_PAT, value=x)
