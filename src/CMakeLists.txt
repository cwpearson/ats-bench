add_subdirectory(common)

add_executable(test-system-alloc ${BENCH_CUDA_SRCS} test_system_alloc.cu)
target_link_libraries(test-system-alloc spdlog::spdlog)
target_link_libraries(test-system-alloc nvidia-ml)

add_subdirectory(bandwidth)