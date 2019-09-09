# ats-bench

Benchmarks for CUDA Address Translation Services

## Topics:

* Raw Bandwidth
  * Copy an array into GPU memory
    - [ ] From CPU (cached)
    - [ ] From CPU (uncached)
* Access contention
  - [ ] Interleave accesses and modifications to strided regions with multiple GPUs

## Building

```
mkdir build
cd buid
cmake ..
make
```

## Running

Benchmarks are in `src/benchmarks/*`. Do `src/benchmark/[class]/[the-benchmark] --help` to see all of the options.

There are also some utilities in `src/`:
* `src/test-system-allocator`: check to see if the system allocator is working


## Running on HAL

* [New Job Queues](https://wiki.ncsa.illinois.edu/display/ISL20/Job+management+with+SLURM#JobmanagementwithSLURM-NewJobQueues)
* [Real-time System Status](https://hal-monitor.ncsa.illinois.edu:3000/)

* Interactive Job (1 GPU)
```
srun --partition=gpu-debug --pty --nodes=1 \
  --ntasks-per-node=12 --cores-per-socket=12 \
  --gres=gpu:v100:1 --mem-per-cpu=1500 \
  --time=2:00:00 --wait=0 --export=ALL /bin/bash
```



## Acks

* Uses [lyra](https://github.com/bfgroup/Lyra) for cli option parsing.
* Uses [hunter](https://github.com/ruslo/hunter) for package management.
* Uses [spdlog](https://github.com/gabime/spdlog) for logging.
