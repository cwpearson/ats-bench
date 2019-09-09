

add_executable(to-gpu ${BENCH_CUDA_SRCS} to_gpu.cu)
target_link_libraries(to-gpu spdlog::spdlog)