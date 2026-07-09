#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                                \
    if (status != cudaSuccess) {                                                \
      std::cerr << #call << " failed: " << cudaGetErrorString(status) << "\n"; \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)

__global__ void set_condition(cudaGraphConditionalHandle handle,
                              const int* predicate) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    cudaGraphSetConditional(handle, *predicate != 0);
  }
}

__global__ void true_branch(int* output) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    output[0] = 111;
    output[1] = 1;
  }
}

__global__ void false_branch(int* output) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    output[0] = 222;
    output[1] = 0;
  }
}

void add_kernel_node(cudaGraph_t graph, void* kernel, void** args) {
  cudaKernelNodeParams params{};
  params.func = kernel;
  params.gridDim = dim3(1);
  params.blockDim = dim3(1);
  params.kernelParams = args;

  cudaGraphNode_t node{};
  CHECK_CUDA(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params));
}

int main() {
  int* d_predicate = nullptr;
  int* d_output = nullptr;
  CHECK_CUDA(cudaMalloc(&d_predicate, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_output, 2 * sizeof(int)));

  cudaStream_t stream{};
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaGraph_t graph{};
  CHECK_CUDA(cudaGraphCreate(&graph, 0));

  cudaGraphConditionalHandle condition{};
  CHECK_CUDA(cudaGraphConditionalHandleCreate(
      &condition, graph, 0, cudaGraphCondAssignDefault));

  void* set_condition_args[] = {&condition, &d_predicate};
  cudaKernelNodeParams set_condition_params{};
  set_condition_params.func = reinterpret_cast<void*>(set_condition);
  set_condition_params.gridDim = dim3(1);
  set_condition_params.blockDim = dim3(1);
  set_condition_params.kernelParams = set_condition_args;

  cudaGraphNode_t set_condition_node{};
  CHECK_CUDA(cudaGraphAddKernelNode(
      &set_condition_node, graph, nullptr, 0, &set_condition_params));

  cudaGraphNodeParams if_params{};
  if_params.type = cudaGraphNodeTypeConditional;
  if_params.conditional.handle = condition;
  if_params.conditional.type = cudaGraphCondTypeIf;
  if_params.conditional.size = 2;

  cudaGraphNode_t if_node{};
  CHECK_CUDA(cudaGraphAddNode(
      &if_node, graph, &set_condition_node, 1, &if_params));

  void* branch_args[] = {&d_output};
  add_kernel_node(if_params.conditional.phGraph_out[0],
                  reinterpret_cast<void*>(true_branch), branch_args);
  add_kernel_node(if_params.conditional.phGraph_out[1],
                  reinterpret_cast<void*>(false_branch), branch_args);

  cudaGraphExec_t graph_exec{};
  CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, 0));

  for (int predicate : {1, 0}) {
    int host_output[2] = {-1, -1};

    CHECK_CUDA(cudaMemcpyAsync(d_predicate, &predicate, sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemsetAsync(d_output, 0xff, 2 * sizeof(int), stream));
    CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
    CHECK_CUDA(cudaMemcpyAsync(host_output, d_output, 2 * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::cout << "predicate=" << predicate << " -> output=("
              << host_output[0] << ", " << host_output[1] << ")\n";
  }

  CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
  CHECK_CUDA(cudaGraphDestroy(graph));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_predicate));
  return 0;
}
