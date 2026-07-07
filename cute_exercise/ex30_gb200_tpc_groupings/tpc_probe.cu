#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _cuda_status = (expr);                                         \
    if (_cuda_status != cudaSuccess) {                                         \
      std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(_cuda_status));                          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Row {
  int cluster_size;
  int iteration;
  int block_idx;
  int cluster_id;
  int cluster_rank;
  int smid;
};

extern "C" __global__ void record_smid_rows(Row* rows, int iteration,
                                             int cluster_size,
                                             unsigned long long spin_cycles) {
  extern __shared__ unsigned char smem[];
  if (threadIdx.x == 0) {
    smem[0] = static_cast<unsigned char>(blockIdx.x);
  }
  __syncthreads();

  unsigned int smid = 0;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  int cluster_rank = cluster.block_rank();
  int row_idx = iteration * gridDim.x + blockIdx.x;

  if (threadIdx.x == 0) {
    rows[row_idx] = {
        cluster_size,
        iteration,
        static_cast<int>(blockIdx.x),
        static_cast<int>(blockIdx.x) / cluster_size,
        cluster_rank,
        static_cast<int>(smid),
    };
  }

  unsigned long long start = clock64();
  while (clock64() - start < spin_cycles) {
    asm volatile("");
  }
}

struct Options {
  int cluster_size = 1;
  int iters = 10;
  int blocks = 0;
  int threads = 32;
  int dynamic_smem = 0;
  unsigned long long spin_cycles = 1000;
};

int parse_int(const char* value, const char* name) {
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (*value == '\0' || *end != '\0' || parsed < 0) {
    std::fprintf(stderr, "invalid %s: %s\n", name, value);
    std::exit(2);
  }
  return static_cast<int>(parsed);
}

unsigned long long parse_ull(const char* value, const char* name) {
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 10);
  if (*value == '\0' || *end != '\0') {
    std::fprintf(stderr, "invalid %s: %s\n", name, value);
    std::exit(2);
  }
  return parsed;
}

Options parse_options(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--cluster-size") == 0 && i + 1 < argc) {
      opts.cluster_size = parse_int(argv[++i], "--cluster-size");
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      opts.iters = parse_int(argv[++i], "--iters");
    } else if (std::strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
      opts.blocks = parse_int(argv[++i], "--blocks");
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      opts.threads = parse_int(argv[++i], "--threads");
    } else if (std::strcmp(argv[i], "--dynamic-smem") == 0 && i + 1 < argc) {
      opts.dynamic_smem = parse_int(argv[++i], "--dynamic-smem");
    } else if (std::strcmp(argv[i], "--spin-cycles") == 0 && i + 1 < argc) {
      opts.spin_cycles = parse_ull(argv[++i], "--spin-cycles");
    } else {
      std::fprintf(
          stderr,
          "usage: %s --cluster-size N [--iters N] [--blocks N] "
          "[--threads N] [--dynamic-smem BYTES] [--spin-cycles N]\n",
          argv[0]);
      std::exit(2);
    }
  }

  if (opts.cluster_size < 1 || opts.iters < 1 || opts.threads < 1) {
    std::fprintf(stderr, "cluster size, iters, and threads must be positive\n");
    std::exit(2);
  }
  return opts;
}

int device_attr(cudaDeviceAttr attr, int device, int fallback = 0) {
  int value = fallback;
  cudaError_t status = cudaDeviceGetAttribute(&value, attr, device);
  if (status == cudaErrorInvalidValue) {
    cudaGetLastError();
    return fallback;
  }
  CUDA_CHECK(status);
  return value;
}

void print_json_string(const char* value) {
  std::putchar('"');
  for (const char* p = value; *p != '\0'; ++p) {
    if (*p == '"' || *p == '\\') {
      std::putchar('\\');
    }
    std::putchar(*p);
  }
  std::putchar('"');
}

int main(int argc, char** argv) {
  Options opts = parse_options(argc, argv);

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  int blocks = opts.blocks;
  if (blocks == 0) {
    blocks = (prop.multiProcessorCount / opts.cluster_size) * opts.cluster_size;
  }
  if (blocks < opts.cluster_size || blocks % opts.cluster_size != 0) {
    std::fprintf(stderr, "--blocks must be a positive multiple of cluster size\n");
    return 2;
  }

  int dynamic_smem = opts.dynamic_smem;
  if (dynamic_smem == 0) {
    int optin = prop.sharedMemPerBlockOptin;
    if (optin == 0) {
      optin = prop.sharedMemPerBlock;
    }
    dynamic_smem =
        std::min(optin, static_cast<int>(prop.sharedMemPerMultiprocessor));
  }

  CUDA_CHECK(cudaFuncSetAttribute(
      record_smid_rows, cudaFuncAttributeMaxDynamicSharedMemorySize,
      dynamic_smem));
  CUDA_CHECK(cudaFuncSetAttribute(
      record_smid_rows, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  Row* device_rows = nullptr;
  size_t row_count = static_cast<size_t>(opts.iters) * blocks;
  CUDA_CHECK(cudaMalloc(&device_rows, row_count * sizeof(Row)));

  dim3 grid(blocks, 1, 1);
  dim3 block(opts.threads, 1, 1);
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = opts.cluster_size;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = dynamic_smem;
  config.numAttrs = 1;
  config.attrs = attrs;

  for (int iter = 0; iter < opts.iters; ++iter) {
    CUDA_CHECK(cudaLaunchKernelEx(&config, record_smid_rows, device_rows, iter,
                                  opts.cluster_size, opts.spin_cycles));
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<Row> host_rows(row_count);
  CUDA_CHECK(cudaMemcpy(host_rows.data(), device_rows, row_count * sizeof(Row),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_rows));

  int runtime_version = 0;
  int driver_version = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));

  std::printf("{\"device\":{\"name\":");
  print_json_string(prop.name);
  std::printf(
      ",\"compute_capability\":\"%d.%d\",\"multi_processor_count\":%d,"
      "\"runtime_version\":%d,\"driver_version\":%d,"
      "\"shared_mem_per_block_optin\":%zu,"
      "\"shared_mem_per_multiprocessor\":%zu},",
      prop.major, prop.minor, prop.multiProcessorCount, runtime_version,
      driver_version, static_cast<size_t>(prop.sharedMemPerBlockOptin),
      static_cast<size_t>(prop.sharedMemPerMultiprocessor));
  std::printf(
      "\"launch\":{\"cluster_size\":%d,\"iters\":%d,\"blocks\":%d,"
      "\"threads\":%d,\"dynamic_smem\":%d,\"spin_cycles\":%llu},",
      opts.cluster_size, opts.iters, blocks, opts.threads, dynamic_smem,
      opts.spin_cycles);
  std::printf("\"rows\":[");
  for (size_t i = 0; i < host_rows.size(); ++i) {
    const Row& row = host_rows[i];
    std::printf(
        "%s{\"cluster_size\":%d,\"iteration\":%d,\"block_idx\":%d,"
        "\"cluster_id\":%d,\"cluster_rank\":%d,\"smid\":%d}",
        i == 0 ? "" : ",", row.cluster_size, row.iteration, row.block_idx,
        row.cluster_id, row.cluster_rank, row.smid);
  }
  std::printf("]}\n");
  return 0;
}
