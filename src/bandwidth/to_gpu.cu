#include <iostream>
#include <lyra/lyra.hpp>

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
  bool no_ats_check = false;
  size_t n = 0;

  auto cli =
      lyra::help(help) |
      lyra::opt(debug)["--debug"]("print debug messages to stderr") |
      lyra::opt(verbose)["--verbose"]("print verbose messages to stderr") |
      lyra::opt(no_ats_check)["--no-ats-check"]("skip test for ats") |
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

  if (!no_ats_check) {
    if (test_system_allocator()) {
      LOG(info, "CUDA supports system allocator");
    } else {
      LOG(critical, "CUDA does not work with the system allocator");
      exit(EXIT_FAILURE);
    }
  }

  WithPerformance performance;

  return 0;
}