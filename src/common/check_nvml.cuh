#pragma once

#include <cstdio>

#include <nvml.h>

inline void checkNvml(nvmlReturn_t result, const char *file, const int line) {
  if (result != NVML_SUCCESS) {
    printf("%s@%d: NVML Error: %s\n", file, line,
    nvmlErrorString(result));
    exit(-1);
  }
}

#define NVML(stmt) checkNvml(stmt, __FILE__, __LINE__);