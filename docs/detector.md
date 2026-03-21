# PatternDetector & MatchResult

`PatternDetector` applies a `Pattern` against a specific subgraph of an ONNX model and returns a `MatchResult` on success. `MatchResult` is the dataclass that carries the full result of a successful match.

---

## PatternDetector

### Constructor

```python
PatternDetector(model, start_node=None, end_node=None)
```

| Parameter | Type | Description |
|---|---|---|
| `model` | `ModelProto \| _GraphShim` | The ONNX model or an internal graph shim (from `ONNXQuery`) |
| `start_node` | `str \| NodeProto \| None` | Root of the subgraph to match (inclusive) |
| `end_node` | `str \| NodeProto \| None` | Exclusive upstream boundary (see below) |

Nodes can be identified by name string or `NodeProto` object. Passing an unknown name raises `ValueError`.

For bulk matching across many nodes, prefer `ONNXQuery.match_results(pattern)` which constructs detectors internally and shares the output→node map across all candidates.

```python
from onnx_toolkit import ONNXParser, Pattern, PatternDetector

parser = ONNXParser("model.onnx")
x      = Pattern.any().capture("x")
det    = PatternDetector(parser.model, start_node="Mul_42")
result = det.match(Pattern.silu(x))
```

---

### End-Node Contract

`end_node` defines an **exclusive upstream boundary**. When the DFS would step into a parent node that is exactly `end_node`, that branch is immediately satisfied without recursing into `end_node`. The matched subgraph always consists of nodes between `start_node` (inclusive) and `end_node` (exclusive).

```
[end_node] ──► [A] ──► [B] ──► [start_node]

Matched subgraph = { start_node, B, A }
end_node is NOT part of the match
```

Use `end_node` when you want to constrain the match to a specific region of the graph and prevent it from traversing further upstream into areas you do not care about.

```python
# Match the SiLU pattern starting at Mul_7,
# but stop at Add_3 — don't recurse beyond it
det = PatternDetector(parser.model, start_node="Mul_7", end_node="Add_3")
r   = det.match(Pattern.silu(Pattern.any()))

if r:
    print("Matched. End node:", r.end.name)  # "Add_3"
```

When `end_node` is `None` (the default), the DFS continues until it reaches graph inputs or initializer edges.

---

### `match(pattern: Pattern) → MatchResult | None`

Try to match `pattern` against the subgraph rooted at `start_node`.

Returns a `MatchResult` on success. Returns `None` if the pattern did not match.

```python
result = det.match(Pattern.gelu(Pattern.any()))

if result is None:
    print("No match")
else:
    print("Matched subgraph:", [n.name for n in result.nodes])
    print("Terminal node:", result.end.name)
```

**`result.end` behaviour:**

- If `end_node` was provided and reached during DFS → `result.end` is `end_node`.
- If no `end_node` was provided (match terminated at a leaf) → `result.end` is `start_node`.

---

### `find_all(pattern: Pattern) → list[MatchResult]`

Scan nodes for subgraph matches and return one `MatchResult` per matching start node.

- If `start_node` was provided at construction time: only descendants of that node are scanned.
- Otherwise: the entire graph is scanned.

The output→node map is built once and shared across all candidate tests.

```python
det     = PatternDetector(parser.model)   # no start_node → scan all
results = det.find_all(Pattern.gelu(Pattern.any()))

for r in results:
    print(r.start.name, "→", r.end.name)
```

---

## MatchResult

`MatchResult` is a dataclass returned by a successful `PatternDetector.match()` or produced by `ONNXQuery.match_results()`.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `start` | `NodeProto` | The node at which the match began (the `start_node` given to `PatternDetector`) |
| `end` | `NodeProto` | The terminal boundary node (see above) |
| `nodes` | `list[NodeProto]` | Every node visited during the DFS, in visit order, deduplicated |
| `bindings` | `dict[str, NodeProto]` | Named captures: maps each `.capture("name")` label to the `NodeProto` that matched it |

```python
x      = Pattern.any().capture("input")
gate   = Pattern.op("Sigmoid").capture("sigmoid_gate")
p      = x * gate

det    = PatternDetector(parser.model, start_node="Mul_7")
result = det.match(p)

if result:
    print("start:", result.start.name)
    print("end:  ", result.end.name)
    print("nodes:", [n.name for n in result.nodes])
    print("input node:", result.bindings["input"].name)
    print("gate node: ", result.bindings["sigmoid_gate"].name)
```

---

### `as_query() → ONNXQuery`

Convert the matched subgraph into an `ONNXQuery` for further analysis or export.

Weight tensors are included **only** when the `MatchResult` was produced by `ONNXQuery.match_results()` (which automatically attaches the parser's `tensor_map`). Results from a bare `PatternDetector.match()` call will have an empty tensor map.

```python
# Via ONNXQuery.match_results (recommended — includes weight tensors)
for r in parser.find().match_results(Pattern.gelu(x)):
    sub = r.as_query()
    print(sub.op_types())    # ["Div", "Erf", "Add", "Mul", "Mul"]
    sub.to_dot()             # visualise the matched subgraph

# Via PatternDetector.match directly (no weight tensors)
r   = det.match(Pattern.gelu(x))
sub = r.as_query()           # tensor_map will be empty
```

---

## How the DFS Works

Understanding the matching algorithm helps write correct patterns and diagnose unexpected non-matches.

1. **Start at `start_node`.** The pattern root is matched against this node.
2. **Check end-node boundary.** If the current node is `end_node`, the branch succeeds immediately.
3. **Dispatch on pattern type:**
   - `any_of` — try each alternative; restore state on failure.
   - `const(v)` — check initializer inputs and `Constant` op nodes.
   - `any` (wildcard) — succeeds immediately (after shape/dtype checks).
   - Named op — check `op_type`, then attribute constraints, then shape/dtype constraints.
4. **Separate constant inputs.** Pattern inputs that are `const(v)` are matched directly against the node's initializer edges. Remaining (structural) patterns are matched against parent nodes.
5. **Match parent nodes.** For `Add` and `Mul` (commutative), all permutations of parents are tried. For all other ops, parents are matched in input order.
6. **Backtracking.** If any branch fails, `bindings` and the `trail` (visited node list) are restored to their state before that branch was attempted.

### Trail deduplication

A node that is reachable via multiple input paths (e.g. `x` feeding both sides of `x * sigmoid(x)`) is recorded only once in `result.nodes`. Identity (`is`) comparison is used, not name equality.

### Visited set

The DFS tracks a `visited` frozen set of node names to prevent re-entering nodes already on the current path, which would cause incorrect matches in diamond-shaped subgraphs.

---

## Using PatternDetector vs ONNXQuery

| Use case | Recommended approach |
|---|---|
| Match against a single known node | `PatternDetector(model, start_node="name").match(pattern)` |
| Match against all nodes in the graph | `parser.find().match_results(pattern)` |
| Match within a filtered subset | `parser.find().find_by_op_type("Mul").match_results(pattern)` |
| Scan descendants of a specific node | `PatternDetector(model, start_node="node").find_all(pattern)` |

`ONNXQuery.match_results()` is preferred for bulk matching because it shares the output→node map across all candidates, avoiding repeated rebuilds.
