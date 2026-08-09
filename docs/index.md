# Neuron Toolkit — Documentation

> A framework for inspecting, querying, pattern-matching, and rewriting neural-network computation graphs (ONNX & TFLite).

---

## 📚 Table of Contents

| Guide | What you'll learn |
|---|---|
| [01 — Getting Started](./01_getting_started.md) | Load a model, inspect nodes, tensors, shape info |
| [02 — Query Engine](./02_query_engine.md) | Filter, traverse, sort, and set-operate on nodes |
| [03 — Pattern DSL](./03_pattern_dsl.md) | Build patterns with ops, consts, wildcards, arithmetic |
| [04 — Pattern Matching](./04_pattern_matching.md) | Detect patterns, captures, commutative rules |
| [05 — Graph Rewriting](./05_graph_rewriting.md) | Replace, delete, insert nodes; build new models |
| [06 — Built-in Passes](./06_passes.md) | Conv+BN fusion and `register_initializer` |
| [07 — Graph Slicing](./07_slicing.md) | Extract sub-graphs, single nodes, save & reload |
| [08 — PyTorch Comparison](./08_pytorch_comparison.md) | `compare_outputs` and `@verify_against_pytorch` |
| [09 — TFLite Backend](./09_tflite_backend.md) | Load TFLite models, query, rewrite, register tensors |
| [10 — Multi-Backend & API Reference](./10_api_reference.md) | Full API surface, edge cases, gotchas |
