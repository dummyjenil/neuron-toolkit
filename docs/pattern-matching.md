# Symbolic Pattern Matching DSL & `PatternDetector`

The **Pattern DSL** in `neuron-toolkit` allows you to express declarative graph patterns and subgraphs. It features wildcard matching, constant value validation, attribute constraints, arithmetic operator overloads, activation constructors, and automatic handling of commutative operator permutations.

---

## Pattern Constructors (`Pattern`)

### Core Factories

```python
from neuron_toolkit import Pattern

# Match a specific operator with inputs
p_conv = Pattern.op("Conv", Pattern.any(), Pattern.const(0.5))

# Match any single node (wildcard)
p_any = Pattern.any()

# Match constant/initializer tensor with specific value
p_const = Pattern.const(1.0)

# Match any of multiple alternative pattern branches
p_activation = Pattern.any_of(Pattern.op("Relu"), Pattern.op("Sigmoid"))
```

---

## Pattern Constraints & Captures

- **`.capture(name: str)`**: Assigns a binding name to capture the matched graph node into `MatchResult.bindings[name]`.
- **`.where(**constraints)`**: Enforces attribute constraints on the node.
- **`.with_output_rank(rank: int)`**: Constrains the node output tensor to have a specific rank.
- **`.with_dtype(dtype: str)`**: Constrains the node output tensor to have a specific data type.
- **`.then(op_type_or_pattern)`**: Chains pattern forward, consuming current pattern output as input.

### Example

```python
# Capture Conv node and constrain group attribute to 1
conv_pat = Pattern.op("Conv").capture("my_conv").where(group=1).with_output_rank(4)
```

---

## Arithmetic Operator Overloads

`Pattern` overrides standard Python mathematical operators to make pattern construction concise:

- **Addition**: `p1 + p2` (matches `Pattern.op("Add", p1, p2)`)
- **Subtraction**: `p1 - p2` (matches `Pattern.op("Sub", p1, p2)`)
- **Multiplication**: `p1 * p2` (matches `Pattern.op("Mul", p1, p2)`)
- **Division**: `p1 / p2` (matches `Pattern.op("Div", p1, p2)`)
- **Power**: `p1 ** p2` (matches `Pattern.op("Pow", p1, p2)`)
- **Negation**: `-p1` (matches `Pattern.op("Neg", p1)`)

---

## Built-In Activation Pattern Mixins

The `Pattern` class includes `ActivationMixin`, providing factory methods for over 20 standard neural network activation functions:

| Method | Pattern Representation | Description |
| :--- | :--- | :--- |
| `Pattern.relu(x)` | `Pattern.op("Relu", x)` | ReLU activation |
| `Pattern.sigmoid(x)` | `Pattern.op("Sigmoid", x)` | Sigmoid activation |
| `Pattern.tanh(x)` | `Pattern.op("Tanh", x)` | Tanh activation |
| `Pattern.leaky_relu(x)` | `Pattern.op("LeakyRelu", x)` | Leaky ReLU activation |
| `Pattern.elu(x)` | `Pattern.op("Elu", x)` | ELU activation |
| `Pattern.selu(x)` | `Pattern.op("Selu", x)` | SELU activation |
| `Pattern.softplus(x)` | `Pattern.op("Softplus", x)` | Softplus activation |
| `Pattern.softsign(x)` | `Pattern.op("Softsign", x)` | Softsign activation |
| `Pattern.hardsigmoid(x)` | `Pattern.op("HardSigmoid", x)` | HardSigmoid activation |
| `Pattern.hardswish(x)` | `x * Pattern.hardsigmoid(x)` | HardSwish activation |
| `Pattern.silu(x)` / `.swish(x)` | `x * Pattern.sigmoid(x)` | SiLU / Swish activation |
| `Pattern.gelu(x)` | `x * (Erf(x / 1.414...) + 1.0) * 0.5` | Standard GeLU approximation |
| `Pattern.gelu_tanh(x)` | `0.5 * x * (1.0 + Tanh(...))` | Tanh GeLU approximation |
| `Pattern.mish(x)` | `x * Tanh(Softplus(x))` | Mish activation |
| `Pattern.relu6(x)` | `Pattern.op("Clip", x, 0.0, 6.0)` | ReLU6 activation |
| `Pattern.softmax(x)` | `Pattern.op("Softmax", x)` | Softmax activation |
| `Pattern.log_softmax(x)` | `Pattern.op("LogSoftmax", x)` | LogSoftmax activation |
| `Pattern.prelu(x, slope)` | `Pattern.op("PRelu", x, slope)` | PReLU activation |
| `Pattern.thresholded_relu(x)`| `Pattern.op("ThresholdedRelu", x)` | Thresholded ReLU activation |

---

## Pattern Detection Engine (`PatternDetector`)

`PatternDetector` executes pattern matching over target model graphs:

- **Recursive DFS with Backtracking**: Explores pattern trees while preventing infinite recursion and maintaining trail history.
- **Commutative Operator Permutations**: For operators such as `Add` and `Mul`, `PatternDetector` tests all input permutations to match patterns regardless of operand order.
- **Referential Consistency**: Enforces that shared pattern instances map to identical graph node references.

---

## `MatchResult` Class

When a pattern matches, `PatternDetector` returns a `MatchResult` object:

```python
class MatchResult:
    start: object  # Root node of match
    end: object    # Terminal node of match
    nodes: list[object]  # All matched nodes in subgraph
    bindings: dict[str, object]  # Dict of captured pattern node names
```

### Methods & Properties

- **`.as_query()` / `.query` / `.subgraph`**: Exposes the matched nodes as a `NeuronQuery` for further operations.
