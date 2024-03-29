#pragma once

#include <spdlog/sinks/stdout_color_sinks.h>

#include "check_nvml.cuh"

/*! initialize the benchmark
 */
void init() {
  static bool init_ = false;
  if (init_)
    return;

  // create a logger and implicitly register it
  spdlog::stderr_color_mt("console");

  // init nvml
  NVML(nvmlInit());

  // don't init again if init() called twice
  init_ = true;
}
