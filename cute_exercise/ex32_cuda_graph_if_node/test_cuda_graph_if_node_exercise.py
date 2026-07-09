from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_exercise_documents_cuda_graph_if_nodes() -> None:
    readme = (ROOT / "README.md").read_text()

    assert "cudaGraphCondTypeIf" in readme
    assert "cudaGraphConditionalHandleCreate" in readme
    assert "cudaGraphSetConditional" in readme
    assert "phGraph_out[0]" in readme
    assert "phGraph_out[1]" in readme


def test_demo_uses_a_real_cuda_graph_conditional_node() -> None:
    source = (ROOT / "cuda_graph_if_node_demo.cu").read_text()

    required_tokens = [
        "cudaGraphConditionalHandleCreate",
        "cudaGraphSetConditional",
        "cudaGraphNodeTypeConditional",
        "cudaGraphCondTypeIf",
        "cudaGraphAddNode",
        "phGraph_out[0]",
        "phGraph_out[1]",
    ]

    for token in required_tokens:
        assert token in source


def test_runner_builds_the_demo_source() -> None:
    runner = (ROOT / "run_demo.py").read_text()

    assert "cuda_graph_if_node_demo.cu" in runner
    assert "nvcc" in runner
    assert "--run" in runner


def test_pytorch_demo_uses_conditional_cuda_graph_capture() -> None:
    source = (ROOT / "pytorch_if_node_demo.py").read_text()

    required_tokens = [
        "torch.cond",
        "CUDAGraphCaptureControlFlowOpDispatchMode",
        "begin_capture_to_if_node",
        "end_capture_to_conditional_node",
        "g.replay()",
    ]

    for token in required_tokens:
        assert token in source
