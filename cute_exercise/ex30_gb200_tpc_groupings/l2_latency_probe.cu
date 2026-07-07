#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
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

extern "C" __global__ void l2_latency_signature_kernel(
    const uint32_t* __restrict__ next_nodes,
    unsigned long long* __restrict__ bucket_sums, int* __restrict__ smids,
    uint32_t* __restrict__ final_nodes, int buckets, int samples_per_bucket,
    int stride_words, int warmup_steps, int measure_steps, int target_smid,
    int output_row) {
  extern __shared__ unsigned char smem[];
  if (threadIdx.x == 0) {
    smem[0] = static_cast<unsigned char>(blockIdx.x);
  }
  __syncthreads();

  if (threadIdx.x != 0) {
    return;
  }

  unsigned int smid = 0;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (target_smid >= 0 && static_cast<int>(smid) != target_smid) {
    return;
  }
  int row = target_smid >= 0 ? output_row : static_cast<int>(blockIdx.x);
  smids[row] = static_cast<int>(smid);

  uint32_t current = 0;

  for (int i = 0; i < warmup_steps; ++i) {
    const uint32_t* ptr = next_nodes + size_t(current) * stride_words;
    uint32_t loaded = 0;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(loaded) : "l"(ptr) : "memory");
    current = loaded;
  }

  unsigned long long* sums = bucket_sums + size_t(row) * buckets;
  for (int bucket = 0; bucket < buckets; ++bucket) {
    unsigned long long start = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) :: "memory");
    for (int sample = 0; sample < samples_per_bucket; ++sample) {
      const uint32_t* ptr = next_nodes + size_t(current) * stride_words;
      uint32_t loaded = 0;
      asm volatile("ld.global.cg.u32 %0, [%1];"
                   : "=r"(loaded)
                   : "l"(ptr)
                   : "memory");
      current = loaded;
      asm volatile("" ::"r"(current) : "memory");
    }
    unsigned long long stop = 0;
    asm volatile("" ::"r"(current) : "memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop) :: "memory");
    sums[bucket] = stop - start;
  }
  final_nodes[row] = current;
}

struct Options {
  int buckets = 4096;
  int samples_per_bucket = 16;
  int stride_words = 256;
  int blocks = 0;
  int threads = 32;
  int dynamic_smem = 0;
  int warmup_rounds = 1;
  bool concurrent = false;
  uint64_t seed = 1;
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

uint64_t parse_u64(const char* value, const char* name) {
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 10);
  if (*value == '\0' || *end != '\0') {
    std::fprintf(stderr, "invalid %s: %s\n", name, value);
    std::exit(2);
  }
  return static_cast<uint64_t>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--buckets") == 0 && i + 1 < argc) {
      opts.buckets = parse_int(argv[++i], "--buckets");
    } else if (std::strcmp(argv[i], "--samples-per-bucket") == 0 && i + 1 < argc) {
      opts.samples_per_bucket = parse_int(argv[++i], "--samples-per-bucket");
    } else if (std::strcmp(argv[i], "--stride-words") == 0 && i + 1 < argc) {
      opts.stride_words = parse_int(argv[++i], "--stride-words");
    } else if (std::strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
      opts.blocks = parse_int(argv[++i], "--blocks");
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      opts.threads = parse_int(argv[++i], "--threads");
    } else if (std::strcmp(argv[i], "--dynamic-smem") == 0 && i + 1 < argc) {
      opts.dynamic_smem = parse_int(argv[++i], "--dynamic-smem");
    } else if (std::strcmp(argv[i], "--warmup-rounds") == 0 && i + 1 < argc) {
      opts.warmup_rounds = parse_int(argv[++i], "--warmup-rounds");
    } else if (std::strcmp(argv[i], "--concurrent") == 0) {
      opts.concurrent = true;
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      opts.seed = parse_u64(argv[++i], "--seed");
    } else {
      std::fprintf(
          stderr,
          "usage: %s [--buckets N] [--samples-per-bucket N] "
          "[--stride-words N] [--blocks N] [--threads N] "
          "[--dynamic-smem BYTES] [--warmup-rounds N] [--concurrent] "
          "[--seed N]\n",
          argv[0]);
      std::exit(2);
    }
  }
  if (opts.buckets < 1 || opts.samples_per_bucket < 1 ||
      opts.stride_words < 1 || opts.threads < 1 || opts.warmup_rounds < 0) {
    std::fprintf(stderr, "probe dimensions must be positive\n");
    std::exit(2);
  }
  return opts;
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

  int blocks = opts.blocks == 0 ? prop.multiProcessorCount : opts.blocks;
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
      l2_latency_signature_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      dynamic_smem));

  size_t node_count = size_t(opts.buckets) * opts.samples_per_bucket;
  size_t buffer_words = node_count * opts.stride_words;
  std::vector<uint32_t> host_next(buffer_words, 0);
  std::vector<uint32_t> permutation(node_count);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::mt19937_64 rng(opts.seed);
  std::shuffle(permutation.begin(), permutation.end(), rng);
  for (size_t i = 0; i < node_count; ++i) {
    uint32_t node = permutation[i];
    uint32_t next = permutation[(i + 1) % node_count];
    host_next[size_t(node) * opts.stride_words] = next;
  }

  uint32_t* device_next = nullptr;
  unsigned long long* device_sums = nullptr;
  int* device_smids = nullptr;
  uint32_t* device_final_nodes = nullptr;
  CUDA_CHECK(cudaMalloc(&device_next, buffer_words * sizeof(uint32_t)));
  int rows = prop.multiProcessorCount;
  CUDA_CHECK(cudaMalloc(&device_sums,
                        size_t(rows) * opts.buckets *
                            sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&device_smids, size_t(rows) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&device_final_nodes, size_t(rows) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(device_next, host_next.data(),
                        buffer_words * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(device_sums, 0,
                        size_t(rows) * opts.buckets *
                            sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(device_smids, 0xff, size_t(rows) * sizeof(int)));

  int measure_steps = static_cast<int>(node_count);
  int warmup_steps = measure_steps * opts.warmup_rounds;
  int target_begin = opts.concurrent ? -1 : 0;
  int target_end = opts.concurrent ? 0 : rows;
  for (int target_smid = target_begin; target_smid < target_end; ++target_smid) {
    int output_row = target_smid;
    l2_latency_signature_kernel<<<dim3(blocks), dim3(opts.threads), dynamic_smem>>>(
        device_next, device_sums, device_smids, device_final_nodes, opts.buckets,
        opts.samples_per_bucket, opts.stride_words, warmup_steps, measure_steps,
        target_smid, output_row);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::vector<unsigned long long> host_sums(size_t(rows) * opts.buckets);
  std::vector<int> host_smids(rows);
  std::vector<uint32_t> host_final_nodes(rows);
  CUDA_CHECK(cudaMemcpy(host_sums.data(), device_sums,
                        host_sums.size() * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_smids.data(), device_smids,
                        host_smids.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_final_nodes.data(), device_final_nodes,
                        host_final_nodes.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(device_final_nodes));
  CUDA_CHECK(cudaFree(device_smids));
  CUDA_CHECK(cudaFree(device_sums));
  CUDA_CHECK(cudaFree(device_next));

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
      "\"launch\":{\"blocks\":%d,\"rows\":%d,\"threads\":%d,\"buckets\":%d,"
      "\"samples_per_bucket\":%d,\"stride_words\":%d,"
      "\"dynamic_smem\":%d,\"warmup_rounds\":%d,\"concurrent\":%s,\"seed\":%llu,"
      "\"working_set_bytes\":%zu},",
      blocks, rows, opts.threads, opts.buckets, opts.samples_per_bucket,
      opts.stride_words, dynamic_smem, opts.warmup_rounds,
      opts.concurrent ? "true" : "false",
      static_cast<unsigned long long>(opts.seed), buffer_words * sizeof(uint32_t));
  std::printf("\"rows\":[");
  for (int block = 0; block < rows; ++block) {
    std::printf("%s{\"block_idx\":%d,\"smid\":%d,\"final_node\":%u,\"signature\":[",
                block == 0 ? "" : ",", block, host_smids[block],
                host_final_nodes[block]);
    for (int bucket = 0; bucket < opts.buckets; ++bucket) {
      double avg =
          double(host_sums[size_t(block) * opts.buckets + bucket]) /
          double(opts.samples_per_bucket);
      std::printf("%s%.3f", bucket == 0 ? "" : ",", avg);
    }
    std::printf("]}");
  }
  std::printf("]}\n");
  return 0;
}
