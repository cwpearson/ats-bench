

add_executable(cpu-to-gpu-memcpy ${BENCH_CUDA_SRCS} cpu_to_gpu_memcpy.cu)
target_link_libraries(cpu-to-gpu-memcpy spdlog::spdlog)
target_link_libraries(cpu-to-gpu-memcpy nvidia-ml)

add_executable(cpu-to-gpu-kernel ${BENCH_CUDA_SRCS} cpu_to_gpu_kernel.cu)
target_link_libraries(cpu-to-gpu-kernel spdlog::spdlog)
target_link_libraries(cpu-to-gpu-kernel nvidia-ml)

add_executable(mgpu-contention ${BENCH_CUDA_SRCS} mgpu_contention.cu)
target_link_libraries(mgpu-contention spdlog::spdlog)
target_link_libraries(mgpu-contention nvidia-ml)