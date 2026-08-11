# Module 05: Pattern Matching DSL & `PatternDetector`

This module explains the symbolic pattern matching DSL (`Pattern`), activation constructors, and the recursive Depth-First Search (DFS) graph matching algorithm with backtracking (`PatternDetector`).

---

## 1. The `Pattern` Domain-Specific Language (`src/neuron_toolkit/pattern/dsl.py`)

A `Pattern` represents a symbolic template of a computation subgraph.

### Constructing Symbolic Patterns:

```python
from neuron_toolkit.pattern import Pattern

# 1. Standard Operator Pattern with Input Templates:
conv = Pattern.op("Conv")
relu = Pattern.op("Relu", conv)

# 2. Wildcards & Constants:
any_node = Pattern.any()                   # Matches any node
const_node = Pattern.const(1.0)            # Matches initializer or constant with value 1.0

# 3. Alternatives (Or Logic):
act = Pattern.any_of(Pattern.op("Relu"), Pattern.op("Clip"))

# 4. Captures & Constraints:
p = (
    Pattern.op("Conv")
    .capture("my_conv")                    # Bind matched node to name 'my_conv'
    .where(group=1)                        # Attribute constraint
    .with_output_rank(4)                   # Tensor rank constraint
    .with_dtype("float32")                 # Data type constraint
)
```

---

## 2. Operator Overloads & Activation Mixin (`_activations.py`)

Patterns support standard Python arithmetic operator overloads:

```python
# Operator Overloading:
p1 = Pattern.op("MatMul")
p2 = Pattern.const(0.5)

p_add = p1 + p2    # Pattern.op("Add", [p1, p2])
p_mul = p1 * p2    # Pattern.op("Mul", [p1, p2])
p_sub = p1 - p2    # Pattern.op("Sub", [p1, p2])
p_div = p1 / p2    # Pattern.op("Div", [p1, p2])
p_pow = p1 ** 3    # Pattern.op("Pow", [p1, Pattern.const(3)])
```

### Activation Helpers:
`ActivationMixin` (`src/neuron_toolkit/pattern/_activations.py`) adds factory methods for standard activations:

```python
x = Pattern.any()

Pattern.relu(x)          # Pattern.op("Relu", x)
Pattern.sigmoid(x)       # Pattern.op("Sigmoid", x)
Pattern.silu(x)          # x * Pattern.sigmoid(x)
Pattern.gelu(x)          # x * (Erf(x / sqrt(2)) + 1.0) * 0.5
Pattern.mish(x)          # x * Tanh(Softplus(x))
```

---

## 3. `PatternDetector` Matching Algorithm (`_matcher.py` & `detector.py`)

`PatternDetector` scans a neural network graph to find subgraphs matching a `Pattern`.

### Key Challenges in Subgraph Matching:
1. **Graphs are Directed Acyclic Graphs (DAGs)**, not trees. Nodes can branch and merge.
2. **Referential Consistency**: If a pattern reuses a sub-pattern node (e.g. $x + x$), both inputs MUST resolve to the exact same node in the model graph.
3. **Commutative Operators**: Operators like `Add` or `Mul` can receive inputs in any order ($A + B = B + A$).
4. **State Backtracking**: If a candidate branch fails halfway through, all node captures and memoization MUST be cleanly reverted.

---

## 4. Deep Dive into `MatchContext` & State Management

Matching state is encapsulated inside `MatchContext` (`src/neuron_toolkit/pattern/_matcher.py`):

```python
@dataclass
class MatchContext:
    detector: PatternDetector
    bindings: dict[str, object] = field(default_factory=dict) # Capture name -> matched node
    trail: list[object] = field(default_factory=list)          # Matched nodes in traversal order
    memo: dict[int, int] = field(default_factory=dict)         # id(Pattern) -> id(Node)
    visited: set[str] = field(default_factory=set)             # Visited node names
```

### Snapshot & Restore (Backtracking):
To support backtracking when an alternative path fails:

```python
def snapshot(self) -> tuple[dict, list, dict, set]:
    return (dict(self.bindings), list(self.trail), dict(self.memo), set(self.visited))

def restore(self, snap: tuple[dict, list, dict, set]) -> None:
    self.bindings, self.trail, self.memo, self.visited = snap
```

---

## 5. Recursive DFS Matcher (`_match_recursive`) Step-by-Step

The matcher works backwards from output nodes to input nodes:

```
Step 1: Check Memoized Referential Consistency (id(Pattern) in ctx.memo)
  │
Step 2: Boundary Check (Node is end_node)
  │
Step 3: Handle Special Pattern Types (__any_of__, __const__, __any__)
  │
Step 4: Check Op Type & Node Attributes & Output Shape/Dtype
  │
Step 5: Process Inputs
  ├── If Op is Commutative ("Add", "Mul"): Run Permutations of Parents
  └── Else: Match Inputs in Exact Positional Order
  │
Step 6: Finalize Match (Record node in trail, store captures in bindings)
```

### Commutative Matching Algorithm (`_match_commutative`):
For `Add` and `Mul` operations:
1. Constant patterns (`__const__`) are matched first against model initializers.
2. For remaining non-constant inputs, `itertools.permutations()` tests all parent permutations against input patterns.
3. If any permutation matches recursively, the match succeeds!

```python
for perm in itertools.permutations(actual_parents, len(non_const_pats)):
    snap = ctx.snapshot()
    if all(self._match_recursive(p, pat, ctx) for p, pat in zip(perm, non_const_pats)):
        return True
    ctx.restore(snap)
```

In the next module, we will explore **Staged Graph Rewriting**.
