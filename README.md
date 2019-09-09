# ats-bench

Benchmarks for CUDA Address Translation Services

## Topics:

* Raw Bandwidth
  * Copy an array into GPU memory
    - [ ] From CPU (cached)
    - [ ] From CPU (uncached)
* Access contention
  - [ ] Interleave accesses and modifications to strided regions with multiple GPUs


## Running on HAL


## Acks

* Uses [lyra](https://github.com/bfgroup/Lyra) for cli option parsing.
* Uses [hunter](https://github.com/ruslo/hunter) for package management.
* Uses [spdlog](https://github.com/gabime/spdlog) for logging.
