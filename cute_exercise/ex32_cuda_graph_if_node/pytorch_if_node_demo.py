import torch
from torch._higher_order_ops.cudagraph_conditional_nodes import (
    CUDAGraphCaptureControlFlowOpDispatchMode,
    ControlFlowOpWarmupDispatchMode,
)


def true_fn(x: torch.Tensor) -> torch.Tensor:
    return x + 10


def false_fn(x: torch.Tensor) -> torch.Tensor:
    return x - 10


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this demo.")

    required_methods = [
        "begin_capture_to_if_node",
        "end_capture_to_conditional_node",
    ]
    missing = [
        name for name in required_methods if not hasattr(torch.cuda.CUDAGraph, name)
    ]
    if missing:
        raise SystemExit(
            "This PyTorch build does not expose CUDA Graph conditional capture: "
            + ", ".join(missing)
        )

    x = torch.ones(4, device="cuda")
    pred = torch.tensor(True, device="cuda")

    # Warm up both branches before real capture. PyTorch's helper does this
    # with a throwaway relaxed capture so branch-local setup does not leak into
    # the captured graph.
    with ControlFlowOpWarmupDispatchMode():
        torch.cond(pred, true_fn, false_fn, (x,))
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(g, stream=capture_stream, capture_error_mode="relaxed"):
        with CUDAGraphCaptureControlFlowOpDispatchMode():
            y_static = torch.cond(pred, true_fn, false_fn, (x,))

    for value in [True, False, True]:
        pred.fill_(value)
        g.replay()
        torch.cuda.synchronize()
        print(f"predicate={value} -> {y_static.tolist()}")


if __name__ == "__main__":
    main()
