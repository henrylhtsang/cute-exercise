# Can CUDA Graphs use `if` / control flow?

Reference:
- <https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaConditionalNodeParams.html>
- <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html>

## Question

Can a CUDA Graph contain control flow, such as an `if` branch, without returning
to the CPU between kernels?

The likely keyword is "IF nodes": in CUDA Runtime API terms, these are
conditional graph nodes using `cudaGraphCondTypeIf`.

## Answer

Yes. CUDA Graphs support conditional nodes. The CUDA Runtime API exposes:

- `cudaGraphConditionalHandleCreate` to create a condition handle owned by a
  graph;
- `cudaGraphSetConditional` to set that handle from device code;
- `cudaGraphNodeTypeConditional` plus `cudaGraphCondTypeIf` to create an IF
  node;
- a CUDA-owned `phGraph_out` array that holds the body graphs for each branch.

For an IF node:

```text
phGraph_out[0] executes when the condition is non-zero.
phGraph_out[1] executes when the condition is zero, if size == 2.
```

The same API family also has `cudaGraphCondTypeWhile` and
`cudaGraphCondTypeSwitch`, so the feature is broader than just IF/ELSE.

## Demo

The demo in `cuda_graph_if_node_demo.cu` builds one graph:

```text
set_condition_kernel
  -> cudaGraphCondTypeIf node
       true body:  write 111
       false body: write 222
```

The condition kernel reads a device integer and calls
`cudaGraphSetConditional(handle, predicate != 0)`. The conditional node then
runs either `phGraph_out[0]` or `phGraph_out[1]` inside the graph launch.

Build only:

```bash
python cute_exercise/ex32_cuda_graph_if_node/run_demo.py
```

Build and run:

```bash
python cute_exercise/ex32_cuda_graph_if_node/run_demo.py --run
```

Expected output:

```text
predicate=1 -> output=(111, 1)
predicate=0 -> output=(222, 0)
```

## Python / PyTorch

There are two Python answers:

1. CUDA Python bindings expose the low-level conditional-node pieces through
   `cuda.bindings.runtime`, including `cudaGraphConditionalHandleCreate`,
   `cudaGraphAddNode`, `cudaConditionalNodeParams`, and enum values such as
   `cudaGraphConditionalNodeType.cudaGraphCondTypeIf`.
2. This PyTorch build has beta conditional CUDA Graph capture hooks:
   `torch.cuda.CUDAGraph.begin_capture_to_if_node` and
   `torch.cuda.CUDAGraph.end_capture_to_conditional_node`.

The runnable PyTorch demo uses `torch.cond` with
`CUDAGraphCaptureControlFlowOpDispatchMode`. It captures a graph once, then
changes a CUDA scalar predicate before each `g.replay()`:

```bash
python cute_exercise/ex32_cuda_graph_if_node/pytorch_if_node_demo.py
```

Expected output:

```text
predicate=True -> [11.0, 11.0, 11.0, 11.0]
predicate=False -> [-9.0, -9.0, -9.0, -9.0]
predicate=True -> [11.0, 11.0, 11.0, 11.0]
```

The key point is that a normal Python `if pred:` is not dynamic inside a
captured CUDA graph. The predicate must be a CUDA scalar tensor used by
`torch.cond`, and the graph capture must use PyTorch's conditional-node capture
mode so the branch becomes graph-level control flow.

## Notes

Conditional-node graph restrictions matter:

- the condition handle must be created before the conditional node;
- the control value is set by the default handle value and/or by
  `cudaGraphSetConditional` in device code;
- racing concurrent calls to `cudaGraphSetConditional` for the same handle are
  undefined behavior;
- graphs containing conditional nodes cannot be used as child nodes, cloned, or
  instantiated more than once at the same time.

This is graph-level control flow. It does not mean arbitrary host-side `if`
statements execute during a graph launch; the branch is represented explicitly
as a CUDA Graph conditional node.
