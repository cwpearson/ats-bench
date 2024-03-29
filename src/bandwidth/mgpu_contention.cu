#include <iostream>
#include <lyra/lyra.hpp>

#include "common/cache_control.hpp"
#include "common/check_cuda.cuh"
#include "common/init.hpp"
#include "common/logger.hpp"
#include "common/perf_control.hpp"
#include "common/string.hpp"
#include "common/test_system_allocator.hpp"

/*!

0 1 2 3 4 5 6 7 8 9 10 11 12

0 0 1 1 2 2 3 3 4 4 5 5 6  6

0 1 0_1 2 3 2_3 4 5 4_5 6  7

0 1 2 3 0_1_2_3 4 5  6  7  4_5_6_7


This kernel does (n / numWorkers) * 1000 updates to memory
*/
__global__ void contention_kernel(volatile char *data, const size_t n,
                                  const size_t stride, const size_t workerId,
                                  const size_t numWorkers) {

  // modify field when its in a chunk chunks % workerId == 0
  for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
       i += gridDim.x * blockDim.x) {
    size_t chunkIdx = (i / stride) * numWorkers + workerId;
    size_t fieldIdx = i % stride;
    size_t dataIdx = chunkIdx * stride + fieldIdx;
    // if (chunkIdx == 0 || chunkIdx == 1) {
    //   printf("%lu %lu %lu %lu\n", i, chunkIdx, fieldIdx, dataIdx);
    // }
    if (dataIdx < n) {
      for (size_t j = 0; j < 1000; ++j) {
        data[dataIdx] += 1;
      }
    } else {
      break;
    }
  }
}

std::string get_header(const std::string &sep, const size_t nIters) {
  std::string result;
  result = "bmark" + sep + "alloc" + sep + "stride" + sep + "gpus";
  for (size_t i = 0; i < nIters; ++i) {
    result += sep + fmt::format("{}", i);
  }
  return result;
}

int main(int argc, char **argv) {
  // init the benchmark library
  init();

  enum AllocMethod { SYSTEM, MANAGED, MAPPED };

  bool debug = false;
  bool headerOnly = false;
  bool help = false;
  bool noAtsCheck = false;
  std::string sep = ",";
  bool strictPerf = false;
  bool verbose = false;

  std::vector<int> gpus;
  int nIters = 5;
  size_t nBytes = 1024 * 1024;
  size_t stride = 32;
  AllocMethod allocMethod = SYSTEM;

  auto cli =
      lyra::help(help) |
      lyra::opt(debug)["--debug"]("print debug messages to stderr") |
      lyra::opt(headerOnly)["--header-only"]("only print header") |
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
            } else if ("mapped" == s) {
              allocMethod = MAPPED;
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

  if (headerOnly) {
    fmt::print("{}\n", get_header(sep, nIters));
    exit(EXIT_SUCCESS);
  }

  if (gpus.empty()) {
    gpus.push_back(0);
  }
  std::string gpuString;
  for (auto gpu : gpus) {
    gpuString += fmt::format("{}", gpu);
  }

#ifndef NDEBUG
  LOG(warn, "this is not a release build.");
#endif

  // test system allocator before any GPU stuff happens
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

  // lock GPU clocks
  WithMaxGPUClocks clockMaxer(gpus, strictPerf);

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
  case MAPPED: {
    CUDA_RUNTIME(cudaHostAlloc(&data, nBytes, cudaHostAllocMapped));
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

  // zero allocation
  // for (size_t i = 0; i < nBytes; ++i) {
  //   data[i] = 0;
  // }

  // create one stream per gpu
  std::vector<cudaStream_t> streams;
  for (const auto gpu : gpus) {
    cudaStream_t stream;
    CUDA_RUNTIME(cudaSetDevice(gpu));
    CUDA_RUNTIME(cudaStreamCreate(&stream));
    streams.push_back(stream);
  }

  std::vector<double> updatesPerSec;
  for (int i = 0; i < nIters; ++i) {

    // run the workload
    auto wct = std::chrono::system_clock::now();
    for (size_t j = 0; j < gpus.size(); ++j) {
      auto gpu = gpus[j];
      auto stream = streams[j];
      LOG(debug, "launch contention_kernel<<<250, 512, 0, {}>>> on {}",
          uintptr_t(stream), gpu);
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

    const size_t numUpdates = nBytes * 1000;
    updatesPerSec.push_back(numUpdates / elapsed);
  }
  std::string allocMethodString;
  if (SYSTEM == allocMethod) {
    allocMethodString += "system";
  } else if (MANAGED == allocMethod) {
    allocMethodString += "managed";
  } else if (MAPPED == allocMethod) {
    allocMethodString += "mapped";
  }
  std::string output =
      fmt::format("{}{}{}{}{}{}{}", "mgpu-contention", sep, allocMethodString,
                  sep, stride, sep, gpuString);
  output += sep + to_string(updatesPerSec);
  fmt::print("{}\n", output);

  // check allocation
  // for (size_t i = 0; i < nBytes; ++i) {
  //   if (data[i] != data[0]) {
  //     LOG(critical, "kernel bug: {}@{} != {}@0", int(data[i]), i,
  //     int(data[0])); exit(EXIT_FAILURE);
  //   }
  // }

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
  case MAPPED: {
    CUDA_RUNTIME(cudaFreeHost(data));
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
