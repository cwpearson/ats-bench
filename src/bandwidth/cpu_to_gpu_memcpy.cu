#include <iostream>
#include <lyra/lyra.hpp>

#include "common/cache_control.hpp"
#include "common/check_cuda.cuh"
#include "common/init.hpp"
#include "common/logger.hpp"
#include "common/perf_control.hpp"
#include "common/test_system_allocator.hpp"

int main(int argc, char **argv) {
  init();

  bool help = false;
  bool debug = false;
  bool verbose = false;
  bool strictPerf = false;
  bool flush = false;
  size_t n = 0;

  auto cli =
      lyra::help(help) |
      lyra::opt(debug)["--debug"]("print debug messages to stderr") |
      lyra::opt(verbose)["--verbose"]("print verbose messages to stderr") |
      lyra::opt(flush)["--flush"]("flush CPU cache") |
      lyra::opt(strictPerf)["--strict-perf"](
          "fail if system performance cannot be controlled") |
      lyra::arg(n, "size")("Size").required();

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

  // set CPU to high performance mode
  WithPerformance performanceGovernor(strictPerf);

  // disable CPU boosting
  WithoutBoost boostDisabler(strictPerf);

  typedef int32_t Type;
  const size_t nElems = n;
  const size_t nBytes = nElems * sizeof(Type);
  Type *dst;
  CUDA_RUNTIME(cudaMalloc(&dst, nBytes));
  Type *src = new Type[n];
  if (!src) {
    LOG(critical, "failed allocation");
    exit(EXIT_FAILURE);
  }

  // touch all src lines
  LOG(info, "CPU touch src allocation");
  const size_t lineSize = cache_linesize();
  LOG(debug, "CPU line size {}", lineSize);
  for (size_t i = 0; i < nElems; i += lineSize / sizeof(Type)) {
    src[i] = 0;
  }

  // flush src pages from cache
  if (flush) {
    LOG(info, "flush CPU cache");
    flush_all(src, nBytes);
  }

  // create stream
  std::vector<cudaStream_t> streams(1);
  CUDA_RUNTIME(cudaStreamCreate(&streams[0]));

  // create event
  cudaEvent_t start, stop;
  CUDA_RUNTIME(cudaEventCreate(&start));
  CUDA_RUNTIME(cudaEventCreate(&stop));

  // copy to GPU
  LOG(info, "operation");
  CUDA_RUNTIME(cudaEventRecord(start, streams[0]));
  CUDA_RUNTIME(
      cudaMemcpyAsync(dst, src, nBytes, cudaMemcpyDefault, streams[0]));
  CUDA_RUNTIME(cudaEventRecord(stop, streams[0]));

  // wait for copy to be done
  CUDA_RUNTIME(cudaEventSynchronize(stop));

  float millis;
  CUDA_RUNTIME(cudaEventElapsedTime(&millis, start, stop));
  double bytesPerSec = (nBytes / millis) * 1e3;
  fmt::print("{} {} {}\n", nBytes, bytesPerSec, millis / 1e3);

  // destroy stream
  for (auto stream : streams) {
    CUDA_RUNTIME(cudaStreamDestroy(stream));
  }

  return 0;
}