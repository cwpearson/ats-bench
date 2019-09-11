# ats-bench

| Branch | Status |
|-|-|
| develop | [![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fcwpearson%2Fats-bench%2Fbadge%3Fref%3Ddevelop&style=flat)](https://actions-badge.atrox.dev/cwpearson/ats-bench/goto?ref=develop) |
|master | [![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fcwpearson%2Fats-bench%2Fbadge%3Fref%3Dmaster&style=flat)](https://actions-badge.atrox.dev/cwpearson/ats-bench/goto?ref=master) |

Benchmarks for unified memory handling on GPU

## Microbenchmarks

* How fast can system data be accessed?
  * System buffer fill
    - [ ] {aligned, unaligned} x {mapped/managed/system}
  * System buffer copy
    - [ ] {aligned, unaligned} x {mapped/managed/system}
* What is the granularity of access
  - Interleave modifications to strided regions with multiple GPUs
    - [ ] {mapped/managed/system}
* Are system atomics supported?
  - [ ] {mapped/managed/system}
* How fast can a memory region be created?
  - [ ] Allocation + {no touch / cpu / gpu / both}

## Benchmarks

- [ ] Triangle Counting
- [ ] GEMM



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
* Uses [Atrox/github-actions-badge](https://github.com/Atrox/github-actions-badge) for Github Actions status badge

## Related Work

### 2014
Landaverde, Raphael, et al. "An investigation of unified memory access performance in cuda." 2014 IEEE High Performance Extreme Computing Conference (HPEC). IEEE, 2014. [pdf](https://ieeexplore.ieee.org/iel7/7027306/7040940/07040988.pdf)

### 2015
Li, Wenqiang, et al. "An evaluation of unified memory technology on nvidia gpus." 2015 15th IEEE/ACM International Symposium on Cluster, Cloud and Grid Computing. IEEE, 2015. [pdf](http://hpc.sjtu.edu.cn/ppmm15_uma.pdf)

