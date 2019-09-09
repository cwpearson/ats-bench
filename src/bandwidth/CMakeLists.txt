

add_executable(cpu-to-gpu-memcpy ${BENCH_CUDA_SRCS} cpu_to_gpu_memcpy.cu)
target_link_libraries(cpu-to-gpu-memcpy spdlog::spdlog)

add_executable(cpu-to-gpu-kernel ${BENCH_CUDA_SRCS} cpu_to_gpu_kernel.cu)
target_link_libraries(cpu-to-gpu-kernel spdlog::spdlog)