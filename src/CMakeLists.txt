add_subdirectory(common)

add_executable(test-system-alloc ${BENCH_CUDA_SRCS} test_system_alloc.cu)
target_link_libraries(test-system-alloc spdlog::spdlog)
target_link_libraries(test-system-alloc nvidia-ml)

add_executable(device-query ${BENCH_CUDA_SRCS} device_query.cu)
target_link_libraries(device-query spdlog::spdlog)
target_link_libraries(device-query nvidia-ml)

add_subdirectory(bandwidth)