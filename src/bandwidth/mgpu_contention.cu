#include <iostream>
#include <lyra/lyra.hpp>

#include "common/cache_control.hpp"
#include "common/check_cuda.cuh"
#include "common/init.hpp"
#include "common/logger.hpp"
#include "common/perf_control.hpp"
#include "common/test_system_allocator.hpp"

__global__ void contention_kernel(volatile char *data, const size_t n,
                                  const size_t stride, const size_t workerId,
                                  const size_t numWorkers) {

  const size_t numChunks = n / stride;

  // modify field when its in a chunk chunks % workerId == 0
  for (size_t i = stride * (threadIdx.x * numWorkers + workerId); i < numChunks;
       i += stride * gridDim.x * blockDim.x * numWorkers) {
    for (size_t j = 0; j < 10000; ++j) {
      data[i] += 1;
    }
  }
}

int main(int argc, char **argv) {
  init();

  enum AllocMethod { SYSTEM, MANAGED };

  bool help = false;
  bool debug = false;
  bool verbose = false;
  bool noAtsCheck = false;
  bool strictPerf = false;

  std::vector<int> gpus;
  int nIters = 5;
  size_t nBytes = 1024 * 1024;
  size_t stride = 32;
  AllocMethod allocMethod = SYSTEM;

  auto cli =
      lyra::help(help) |
      lyra::opt(debug)["--debug"]("print debug messages to stderr") |
      lyra::opt(stride, "bytes")["--stride"]("stride length") |
      lyra::opt(verbose)["--verbose"]("print verbose messages to stderr") |
      lyra::opt(noAtsCheck)["--no-ats-check"]("skip test for ATS") |
      lyra::opt(strictPerf)["--strict-perf"](
          "fail if system performance cannot be controlled") |
      lyra::opt(nIters,
                "iters")["-i"]["--iters"]("number of benchmark iterations") |
      lyra::opt(
          [&](std::string s) {
            if ("system" == s) {
              allocMethod = SYSTEM;
              return lyra::parser_result::ok(lyra::parser_result_type::matched);
            } else if ("managed" == s) {
              allocMethod = MANAGED;
              return lyra::parser_result::ok(lyra::parser_result_type::matched);
            } else {
              return lyra::parser_result::runtimeError(
                  "alloc-method must be system,managed");
            }
          },
          "method")["--alloc-method"](
          "host allocation method (system, managed)") |
      lyra::opt(gpus, "device ids")["-g"]("gpus to use") |
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

  if (gpus.empty()) {
    gpus.push_back(0);
  }

  // test system allocator before any CUDA
  if (allocMethod == SYSTEM && !noAtsCheck) {
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

  char *data = nullptr;
  switch (allocMethod) {
  case SYSTEM: {
    data = new char[nBytes];
    break;
  }
  case MANAGED: {
    CUDA_RUNTIME(cudaMallocManaged(&data, nBytes));
    break;
  }
  default: {
    LOG(error, "unexpected value for hostAllocMethod");
    exit(EXIT_FAILURE);
  }
  }

  if (!data) {
    LOG(critical, "failed allocation");
    exit(EXIT_FAILURE);
  }

  // create one stream per gpu
  std::vector<cudaStream_t> streams;
  for (const auto gpu : gpus) {
    cudaStream_t stream;
    CUDA_RUNTIME(cudaSetDevice(gpu));
    CUDA_RUNTIME(cudaStreamCreate(&stream));
    streams.push_back(stream);
  }

  for (int i = 0; i < nIters; ++i) {

    // run the workload
    LOG(debug, "operation");
    auto wct = std::chrono::system_clock::now();
    for (size_t j = 0; j < gpus.size(); ++j) {
      auto gpu = gpus[j];
      auto stream = streams[j];
      CUDA_RUNTIME(cudaSetDevice(gpu));
      contention_kernel<<<250, 512, 0, stream>>>(data, nBytes, stride, j,
                                                 gpus.size());
      CUDA_RUNTIME(cudaGetLastError());
    }

    // wait for workload to be done
    for (const auto stream : streams) {
      CUDA_RUNTIME(cudaStreamSynchronize(stream));
    }
    auto elapsed = (std::chrono::system_clock::now() - wct).count() / 1e9;

    double bytesPerSec = nBytes / elapsed;
    fmt::print("{} {} {}\n", nBytes, bytesPerSec, elapsed);
  }

  // free memory
  switch (allocMethod) {
  case SYSTEM: {
    delete[] data;
    break;
  }
  case MANAGED: {
    CUDA_RUNTIME(cudaFree(data));
    break;
  }
  default: {
    LOG(error, "unexpected value for allocMethod");
    exit(EXIT_FAILURE);
  }
  }

  // destroy stream
  for (auto stream : streams) {
    CUDA_RUNTIME(cudaStreamDestroy(stream));
  }

  return 0;
}