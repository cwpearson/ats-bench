#include <cstdio>

#include <sys/wait.h>
#include <unistd.h>

#include "check_cuda.cuh"

// write ints
__global__ void write_int(int *a, size_t n) {
  for (int i = threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    a[i] = i;
  }
}

bool test_system_allocator() {

  // test the system allocator in a new process
  pid_t pid = fork(); // create child process
  int status;
  switch (pid) {
  case -1: // error
    perror("fork");
    exit(1);

  /* Test the system allocator by trying a write to a system allocation
     If fails, exit with non-zero. Otherwise exit zero.
     The parent will check the child's exit status to determine if it worked.
  */
  case 0: // child process
  {
    CUDA_RUNTIME(cudaDeviceReset());
    int *a = new int[1024];
    *a = 0;
    write_int<<<512, 512>>>(a, 1024);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaErrorIllegalAddress) {
      // fprintf(stderr, "got illegal address using system allocator\n");
      exit(1);
    }
    CUDA_RUNTIME(err);
    if (700 != a[700]) {
      // fprintf(stderr, "write not visible on host\n");
      exit(1);
    }
    exit(0);
  }

  default: // parent process, pid now contains the child pid
    while (-1 == waitpid(pid, &status, 0))
      ; // wait for child to complete
    if (WIFSIGNALED(status) || WEXITSTATUS(status) != 0) {
      // fprintf(stderr, "test process exited with (%d). disabling\n", status);
      return false;
    } else {
      return true;
    }
  }
}