#include <lyra/lyra.hpp>

#include "common/check_cuda.cuh"
#include "common/init.hpp"
#include "common/logger.hpp"
#include "common/test_system_allocator.hpp"

int main(void) {

  init();

  logger::set_level(logger::DEBUG);

  if (test_system_allocator()) {
    LOG(info, "CUDA supports system allocator");
  } else {
    LOG(warn, "No system allocator");
  }

  return 0;
}