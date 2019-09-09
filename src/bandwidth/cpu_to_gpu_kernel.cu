#include <iostream>
#include <lyra/lyra.hpp>

#include "common/cache_control.hpp"
#include "common/check_cuda.cuh"
#include "common/init.hpp"
#include "common/logger.hpp"
#include "common/perf_control.hpp"
#include "common/test_system_allocator.hpp"

template <typename T>
__device__ void copy_func(char *__restrict__ dst, const char *__restrict__ src,
                          const size_t n) {
  const size_t nElems = n / sizeof(T);

  auto dstP = reinterpret_cast<T *>(dst);
  auto srcP = reinterpret_cast<const T *>(src);

  for (size_t i = 0; i < nElems; i += blockDim.x * gridDim.x) {
    dstP[i] = srcP[i];
  }
}

template <typename T>
__global__ void copy_kernel(char *__restrict__ dst,
                            const char *__restrict__ src, const size_t n) {
  // copy in blocks of sizeof(T)
  copy_func<T>(dst, src, n);

  // copy in blocks of 1
  size_t rem = n - (n / sizeof(T));
  char *dstTail = &dst[n - rem];
  const char *srcTail = &src[n - rem];
  copy_func<char>(dstTail, srcTail, n - rem);
}

int main(int argc, char **argv) {
  init();

  enum HostAllocMethod { SYSTEM, MANAGED };

  bool help = false;
  bool debug = false;
  bool verbose = false;
  bool noAtsCheck = false;
  bool strictPerf = false;
  bool flush = false;
  int nIters = 5;
  size_t nBytes = 1024 * 1024;
  HostAllocMethod hostAllocMethod = SYSTEM;

  auto cli =
      lyra::help(help) |
      lyra::opt(debug)["--debug"]("print debug messages to stderr") |
      lyra::opt(verbose)["--verbose"]("print verbose messages to stderr") |
      lyra::opt(noAtsCheck)["--no-ats-check"]("skip test for ATS") |
      lyra::opt(flush)["--flush"]("flush CPU cache") |
      lyra::opt(strictPerf)["--strict-perf"](
          "fail if system performance cannot be controlled") |
      lyra::opt(nIters,
                "iters")["-i"]["--iters"]("number of benchmark iterations") |
      lyra::opt(
          [&](std::string s) {
            if ("system" == s) {
              hostAllocMethod = SYSTEM;
              return lyra::parser_result::ok(lyra::parser_result_type::matched);
            } else if ("managed" == s) {
              hostAllocMethod = MANAGED;
              return lyra::parser_result::ok(lyra::parser_result_type::matched);
            } else {
              return lyra::parser_result::runtimeError(
                  "alloc-method must be system,managed");
            }
          },
          "method")["--alloc-method"](
          "host allocation method (system, managed)") |
      lyra::arg(nBytes, "size")("Size").required();

  auto result = cli.parse({argc, argv});
  if (!result) {
    LOG(error, "Error in command line: {}", result.errorMessage());
    exit(1);
  }

  if (help) {
    std::cout << cli;
    return 0;
  }

  // set logging level
  if (verbose) {
    logger::set_level(logger::Level::TRACE);
  } else if (debug) {
    logger::set_level(logger::Level::DEBUG);
  } else {
    logger::set_level(logger::Level::INFO);
  }

  // log command line before much else happens
  {
    std::string cmd;
    for (int i = 0; i < argc; ++i) {
      if (i != 0) {
        cmd += " ";
      }
      cmd += argv[i];
    }
    LOG(debug, cmd);
  }

  // test system allocator before any CUDA
  if (hostAllocMethod == SYSTEM && !noAtsCheck) {
    if (test_system_allocator()) {
      LOG(info, "CUDA supports system allocator");
    } else {
      LOG(critical, "CUDA does not work with the system allocator");
      exit(EXIT_FAILURE);
    }
  }

  // set CPU to high performance mode
  WithPerformance performanceGovernor(strictPerf);

  // disable CPU boosting
  WithoutBoost boostDisabler(strictPerf);

  typedef int32_t Type;
  char *dst;
  CUDA_RUNTIME(cudaMalloc(&dst, nBytes));
  char *src = nullptr;
  switch (hostAllocMethod) {
  case SYSTEM: {
    src = new char[nBytes];
    break;
  }
  case MANAGED: {
    CUDA_RUNTIME(cudaMallocManaged(&src, nBytes));
    break;
  }
  default: {
    LOG(error, "unexpected value for hostAllocMethod");
    exit(EXIT_FAILURE);
  }
  }

  if (!src) {
    LOG(critical, "failed allocation");
    exit(EXIT_FAILURE);
  }

  // create stream
  std::vector<cudaStream_t> streams(1);
  CUDA_RUNTIME(cudaStreamCreate(&streams[0]));

  // create event
  cudaEvent_t start, stop;
  CUDA_RUNTIME(cudaEventCreate(&start));
  CUDA_RUNTIME(cudaEventCreate(&stop));

  for (int i = 0; i < nIters; ++i) {
    // touch all src lines
    LOG(debug, "CPU touch src allocation");
    const size_t lineSize = cache_linesize();
    LOG(debug, "CPU line size {}", lineSize);
    for (size_t j = 0; j < nBytes; j += lineSize) {
      src[j] = 0;
    }

    // flush src pages from cache
    if (flush) {
      LOG(debug, "flush CPU cache");
      flush_all(src, nBytes);
    }

    // copy to GPU
    LOG(debug, "operation");
    CUDA_RUNTIME(cudaEventRecord(start, streams[0]));
    copy_kernel<Type><<<250, 512, 0, streams[0]>>>(dst, src, nBytes);
    CUDA_RUNTIME(cudaGetLastError());
    CUDA_RUNTIME(cudaEventRecord(stop, streams[0]));

    // wait for copy to be done
    CUDA_RUNTIME(cudaEventSynchronize(stop));

    float millis;
    CUDA_RUNTIME(cudaEventElapsedTime(&millis, start, stop));
    double bytesPerSec = (nBytes / millis) * 1e3;
    fmt::print("{} {} {}\n", nBytes, bytesPerSec, millis / 1e3);
  }

  // free memory
  switch (hostAllocMethod) {
  case SYSTEM: {
    delete[] src;
    break;
  }
  case MANAGED: {
    CUDA_RUNTIME(cudaFree(src));
    break;
  }
  default: {
    LOG(error, "unexpected value for hostAllocMethod");
    exit(EXIT_FAILURE);
  }
  }

  CUDA_RUNTIME(cudaFree(dst));

  // destroy stream
  for (auto stream : streams) {
    CUDA_RUNTIME(cudaStreamDestroy(stream));
  }

  return 0;
}