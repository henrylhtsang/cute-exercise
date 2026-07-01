#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t status = (expr);                                               \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(status));                                \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__global__ void init_words(uint32_t* input, uint32_t* output, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = size_t(blockDim.x) * gridDim.x;
  for (; i < n; i += stride) {
    input[i] = uint32_t(i * 1664525u + 1013904223u);
    output[i] = 0;
  }
}

__global__ void copy_u32_kernel(const uint32_t* __restrict__ input,
                                uint32_t* __restrict__ output, size_t n,
                                int repeat) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = size_t(blockDim.x) * gridDim.x;
  for (int r = 0; r < repeat; ++r) {
    for (size_t idx = i; idx < n; idx += stride) {
      output[idx] = input[idx];
    }
  }
}

__global__ void copy_v4_u32_kernel(const uint32_t* __restrict__ input,
                                   uint32_t* __restrict__ output, size_t chunks,
                                   int repeat) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = size_t(blockDim.x) * gridDim.x;
  for (int r = 0; r < repeat; ++r) {
    for (size_t idx = i; idx < chunks; idx += stride) {
      const uint32_t* src = input + idx * 4;
      uint32_t* dst = output + idx * 4;
      uint32_t x0, x1, x2, x3;
      asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                   : "l"(src));
      asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(dst),
                   "r"(x0), "r"(x1), "r"(x2), "r"(x3));
    }
  }
}

__global__ void copy_v8_u32_kernel(const uint32_t* __restrict__ input,
                                   uint32_t* __restrict__ output, size_t chunks,
                                   int repeat) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = size_t(blockDim.x) * gridDim.x;
  for (int r = 0; r < repeat; ++r) {
    for (size_t idx = i; idx < chunks; idx += stride) {
      const uint32_t* src = input + idx * 8;
      uint32_t* dst = output + idx * 8;
      uint32_t x0, x1, x2, x3, x4, x5, x6, x7;
      asm volatile(
          "ld.global.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
          : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3), "=r"(x4), "=r"(x5),
            "=r"(x6), "=r"(x7)
          : "l"(src));
      asm volatile(
          "st.global.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"l"(
              dst),
          "r"(x0), "r"(x1), "r"(x2), "r"(x3), "r"(x4), "r"(x5), "r"(x6),
          "r"(x7));
    }
  }
}

__global__ void check_words(const uint32_t* input, const uint32_t* output,
                            size_t n, unsigned long long* errors) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = size_t(blockDim.x) * gridDim.x;
  unsigned long long local_errors = 0;
  for (; i < n; i += stride) {
    if (input[i] != output[i]) {
      ++local_errors;
    }
  }
  if (local_errors != 0) {
    atomicAdd(errors, local_errors);
  }
}

struct Options {
  std::string variant = "all";
  size_t n = 1 << 26;
  int repeat = 20;
  int threads = 256;
};

Options parse_options(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
      opts.variant = argv[++i];
    } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
      opts.n = std::strtoull(argv[++i], nullptr, 10);
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      opts.repeat = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      opts.threads = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr,
                   "usage: %s [--variant all|u32|v4_u32|v8_u32] [--n words] "
                   "[--iters repeat] [--threads threads]\n",
                   argv[0]);
      std::exit(2);
    }
  }
  if (opts.n % 8 != 0) {
    std::fprintf(stderr, "--n must be a multiple of 8 uint32 words\n");
    std::exit(2);
  }
  if (opts.repeat < 1 || opts.threads < 1) {
    std::fprintf(stderr, "--iters and --threads must be positive\n");
    std::exit(2);
  }
  return opts;
}

using KernelLauncher = void (*)(const uint32_t*, uint32_t*, size_t, int, dim3,
                                dim3);

void launch_u32(const uint32_t* input, uint32_t* output, size_t n, int repeat,
                dim3 grid, dim3 block) {
  copy_u32_kernel<<<grid, block>>>(input, output, n, repeat);
}

void launch_v4(const uint32_t* input, uint32_t* output, size_t n, int repeat,
               dim3 grid, dim3 block) {
  copy_v4_u32_kernel<<<grid, block>>>(input, output, n / 4, repeat);
}

void launch_v8(const uint32_t* input, uint32_t* output, size_t n, int repeat,
               dim3 grid, dim3 block) {
  copy_v8_u32_kernel<<<grid, block>>>(input, output, n / 8, repeat);
}

bool run_variant(const char* name, int bytes_per_thread, KernelLauncher launch,
                 const Options& opts, const uint32_t* input, uint32_t* output,
                 unsigned long long* errors, dim3 grid, dim3 block,
                 bool comma) {
  CUDA_CHECK(cudaMemset(output, 0, opts.n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(errors, 0, sizeof(unsigned long long)));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  launch(input, output, opts.n, opts.repeat, grid, block);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());

  check_words<<<grid, block>>>(input, output, opts.n, errors);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned long long host_errors = 0;
  CUDA_CHECK(cudaMemcpy(&host_errors, errors, sizeof(host_errors),
                        cudaMemcpyDeviceToHost));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  double gib = double(opts.n * sizeof(uint32_t)) * double(opts.repeat) /
               (1024.0 * 1024.0 * 1024.0);
  double gib_per_s = gib / (double(ms) / 1000.0);
  std::printf(
      "%s\"%s\":{\"ok\":%s,\"errors\":%llu,\"bytes_per_thread\":%d,"
      "\"kernel_ms\":%.6f,\"load_gib_per_s\":%.3f}",
      comma ? "," : "", name, host_errors == 0 ? "true" : "false",
      host_errors, bytes_per_thread, ms, gib_per_s);
  return host_errors == 0;
}

int main(int argc, char** argv) {
  Options opts = parse_options(argc, argv);
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  uint32_t* input = nullptr;
  uint32_t* output = nullptr;
  unsigned long long* errors = nullptr;
  CUDA_CHECK(cudaMalloc(&input, opts.n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&output, opts.n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&errors, sizeof(unsigned long long)));

  dim3 block(opts.threads);
  dim3 grid((opts.n + opts.threads - 1) / opts.threads);
  grid.x = grid.x > 1024 ? 1024 : grid.x;
  init_words<<<grid, block>>>(input, output, opts.n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  bool ok = true;
  bool emitted = false;
  std::printf("{\"device\":\"%s\",\"n_words\":%zu,\"iters\":%d,\"variants\":{",
              prop.name, opts.n, opts.repeat);
  if (opts.variant == "all" || opts.variant == "u32") {
    ok &= run_variant("u32", 4, launch_u32, opts, input, output, errors, grid,
                      block, emitted);
    emitted = true;
  }
  if (opts.variant == "all" || opts.variant == "v4_u32") {
    ok &= run_variant("v4_u32", 16, launch_v4, opts, input, output, errors,
                      grid, block, emitted);
    emitted = true;
  }
  if (opts.variant == "all" || opts.variant == "v8_u32") {
    ok &= run_variant("v8_u32", 32, launch_v8, opts, input, output, errors,
                      grid, block, emitted);
    emitted = true;
  }
  if (!emitted) {
    std::fprintf(stderr, "unknown variant: %s\n", opts.variant.c_str());
    return 2;
  }
  std::printf("}}\n");

  CUDA_CHECK(cudaFree(errors));
  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(input));
  return ok ? 0 : 1;
}
