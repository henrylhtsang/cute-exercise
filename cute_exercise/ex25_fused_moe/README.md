# How do I write a fused MoE kernel?

## Question

A Mixture-of-Experts (MoE) FFN routes each token to its top-`k` experts and
runs only those experts' weights. The naive implementation is a chain of
separate kernels: gate → top-k → scatter/permute tokens by expert → per-expert
GEMM → un-permute → weighted combine. That launches many kernels and moves the
activation through global memory several times.

So: how much can we fuse, and where does fusion actually help? Specifically:

1. How do we express the routed, variable-sized per-expert GEMMs as a single
   grouped GEMM in CuTe DSL (tokens grouped by expert)?
2. Can the gating/scale multiply and the final combine be fused into the GEMM
   epilogue instead of being separate passes?
3. Where does the token permutation (scatter by expert) belong — a prologue
   in the GEMM, or a standalone scatter kernel feeding a contiguous grouped
   GEMM?

## Setup

Implement a CuTe DSL fused MoE FFN with:

- a grouped GEMM over experts with per-expert token counts;
- routing weights applied in the epilogue;
- a correctness check against a reference PyTorch MoE (gate + top-k + per-expert
  matmul + weighted sum), and a benchmark versus the unfused kernel chain.

## Answer

_TODO: fill in after measurements on GB200._
