# onnx_toolkit Documentation

`onnx_toolkit` is a Python library for loading, querying, pattern-matching, and rewriting ONNX computation graphs. It is built around a small set of tightly integrated components that compose naturally into analysis and transformation pipelines.

---

## Table of Contents

| Document | What it covers |
|---|---|
| [Getting Started](getting_started.md) | Installation, first steps, enabling logging |
| [ONNXParser](parser.md) | Loading models, accessing graph metadata, shape inference |
| [ONNXQuery](query.md) | Filtering, traversal, ordering, export, and set operations |
| [Pattern DSL](pattern.md) | Building structural patterns to describe subgraphs |
| [PatternDetector & MatchResult](detector.md) | Matching patterns against graphs and reading results |
| [GraphRewriter](rewriter.md) | Staging and applying graph mutations |
| [Cookbook](cookbook.md) | End-to-end worked examples for common use cases |

---

## Architecture Overview

```
ONNXParser
│
├── .find()  ──────────────────────────────►  ONNXQuery
│                                             │
│                                             ├── .matches(pattern)        ──► ONNXQuery
│                                             ├── .match_results(pattern)  ──► List[MatchResult]
│                                             ├── .topological_sort()      ──► ONNXQuery
│                                             ├── .apply(fn)               ──► ONNXQuery
│                                             ├── .to_dot()                ──► str
│                                             ├── .to_dataframe()          ──► DataFrame
│                                             └── .to_onnx(path)
│
├── .pattern_detect(pattern, start, end) ──► MatchResult | None
│
└── .rewriter()  ──────────────────────────► GraphRewriter
                                             │
                                             ├── .replace(nodes, new_op, …)
                                             ├── .replace_from_result(result, new_op)
                                             ├── .delete(nodes)
                                             ├── .insert_before(target, new_op, …)
                                             ├── .reset()
                                             └── .build(path?)  ──────────► ModelProto
```

`Pattern` objects are constructed independently of any model and describe the *shape* of a subgraph. `PatternDetector` applies a `Pattern` against actual graph nodes and returns a `MatchResult` on success.

---

## Component Summary

**`ONNXParser`** — loads a `.onnx` file, optionally runs ONNX shape inference, and exposes the graph as a `ONNXQuery`. It is the entry point for all other components.

**`ONNXQuery`** — a lazy, chainable view over a subset of graph nodes. Every filter and traversal method returns a new `ONNXQuery` without mutating the original. Queries can be exported to DOT, pandas DataFrames, and standalone `.onnx` files.

**`Pattern`** — a DSL for describing subgraph structures using Python arithmetic operators. Supports wildcards, constants, named captures, attribute constraints, shape and dtype constraints, and union patterns.

**`PatternDetector`** — runs a depth-first search from a chosen start node and tries to match a `Pattern` upstream. Returns a `MatchResult` containing the visited subgraph and all named captures.

**`MatchResult`** — a dataclass returned on a successful match. Carries the start node, end node, all visited nodes, and a dictionary of named captures. Can be converted back to an `ONNXQuery` via `.as_query()`.

**`GraphRewriter`** — stages graph mutations (replacements, deletions, insertions) without touching the original model. Applies all edits atomically when `.build()` is called.

---

## Quick Example

```python
from onnx_toolkit import ONNXParser, Pattern

parser = ONNXParser("model.onnx")

# Find all depthwise convolutions (group == in_channels)
dw_conv = Pattern.op("Conv").where(group=lambda g: g > 1)
matches = parser.find().match_results(dw_conv)

for r in matches:
    print(r.start.name, "→", [n.name for n in r.nodes])

# Fuse GELU subgraphs into the native ONNX Gelu op
x  = Pattern.any().capture("x")
rw = parser.rewriter()
for r in parser.find().match_results(Pattern.gelu(x)):
    rw.replace_from_result(r, new_op="Gelu")
rw.build("model_fused.onnx")
```
