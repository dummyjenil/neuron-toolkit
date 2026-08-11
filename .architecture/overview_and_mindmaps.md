# Architecture Overview & Mermaid Mindmaps

This document provides high-level system mindmaps, component hierarchy diagrams, and data flow models for the `neuron-toolkit` codebase. It is designed to allow AI Agents and senior engineers to rapidly understand the structural relationships across all subsystems.

---

## 1. System Architecture Hierarchy Mindmap

```mermaid
mindmap
  root((neuron-toolkit))
    Public Entry Point
      NeuronGraph
        Backend Auto-detection
        Model Slicing API
        PyTorch Verification Decorator
        GraphPasses Namespace
    Parsers & Backends
      BaseParser Abstract Class
      ONNXParser
        LazyTensorMap ONNX Initializers
        Shape Inference Pipeline
        Attribute Proto Extractor
      TFLiteParser
        LazyTensorMap TFLite Buffers
        TFLiteNode Mock Abstraction
        Category Option Parsers opt_*.py
        Quantization & Sparsity Metadata
    Fluent Query Engine
      NeuronQuery
        Output Map Indexing
        rustworkx PyDiGraph Rust Engine
        graphlib TopologicalSorter C Engine
        Traversal APIs parents children ancestors descendants
        Set Algebra union intersection difference
    Symbolic Pattern Matching
      Pattern DSL
        Operator Patterns & Wildcards & Constants
        Alternatives any_of
        Activation Helpers relu silu gelu mish
        Arithmetic Operator Overloads
      PatternDetector & Matcher
        MatchContext State
        xxhash C-Level 64-bit Memoization
        scipy maximum_bipartite_matching C Engine
        DFS Graph Traversal
        Backtracking Snapshots
        Memoization & Referential Consistency
        Commutative Permutation Engine
      MatchResult Data Class
    Staged Graph Rewriter
      NeuronRewriter Wrapper
      ONNXRewriter
        NetworkX Topological Re-sorting
        Initializer Registration
      TFLiteRewriter
        FlatBuffer Builder Pipeline
        Table Copying & Option Serializers
```

---

## 2. Model Pipeline & Execution Flow Diagram

```mermaid
graph TD
    A[Input Model File: .onnx / .tflite] --> B[NeuronGraph.load]
    B --> C{Backend Detection}
    
    C -->|ONNX Source| D[ONNXParser]
    C -->|TFLite Source| E[TFLiteParser]
    
    D --> F[_GraphShim / BaseParser]
    E --> F
    
    F --> G[NeuronQuery API]
    F --> H[PatternDetector Engine]
    F --> I[NeuronRewriter Engine]
    
    H -->|Matched Subgraph| J[MatchResult]
    J -->|as_query| G
    J -->|replace_from_result| I
    
    I --> K[Topological Sorting & Rebuilding]
    K --> L[Saved Model Output / Memory Proto]
```

---

## 3. Subgraph Boundary Slicing Flow

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant NeuronGraph
    participant Utils as trace_subgraph_boundaries
    participant NetworkX as DiGraph Engine
    participant Backend as ONNX / TFLite Engine

    Client->>NeuronGraph: slice(start_points, end_points)
    NeuronGraph->>Utils: trace_subgraph_boundaries(nodes, tensor_map, S, E)
    Utils->>NetworkX: Build Bipartite DiGraph (Op Nodes + Tensor Nodes)
    Utils->>NetworkX: Compute Descendants(S) and Ancestors(E)
    Utils->>Utils: SubgraphNodes = Descendants(S) ∩ Ancestors(E)
    Utils->>Utils: Identify Boundary Inputs & External Boundary Outputs
    Utils-->>NeuronGraph: Return (kept_ops, inputs, outputs)
    NeuronGraph->>Backend: Extract / Rebuild Subgraph FlatBuffer or ONNX Proto
    Backend-->>NeuronGraph: Return sliced NeuronGraph instance
    NeuronGraph-->>Client: Sliced NeuronGraph
```

---

## 4. Pattern Search State Machine (DFS & Backtracking)

```mermaid
stateDiagram-v2
    [*] --> StartMatch: Target Node & Pattern Provided
    StartMatch --> MemoCheck: Check id(Pattern) in ctx.memo
    MemoCheck --> MatchMemo: Found in Memo (Check Identity Match)
    MemoCheck --> CheckOpType: Not in Memo
    
    CheckOpType --> Fail: OpType Mismatch / Attr Mismatch
    CheckOpType --> CheckInputs: OpType Matches
    
    state CheckInputs {
        [*] --> BranchCommutative: Op in {Add, Mul}
        [*] --> BranchOrdered: Positional Op
        
        BranchOrdered --> MatchParentsSequential: Loop i in Inputs
        BranchCommutative --> PermuteParents: Test Parent Permutations
    }
    
    CheckInputs --> Backtrack: Child Branch Match Failed
    Backtrack --> RestoreSnapshot: Restore ctx.bindings & ctx.memo
    RestoreSnapshot --> TestNextAlternative: Try next Pattern alternative or permutation
    
    CheckInputs --> FinalizeMatch: All Inputs & Constraints Passed
    FinalizeMatch --> StoreBindings: Add node to ctx.trail & ctx.bindings
    StoreBindings --> [*]: Return True (Match Success)
```
