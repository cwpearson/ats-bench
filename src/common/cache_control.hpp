/*!
Routines for controlling CPU caching
*/

#pragma once

#include <algorithm>
#include <cstdio>
#include <iostream>

#ifdef __linux__
#include <unistd.h>
#endif

// https://gcc.gnu.org/onlinedocs/gcc/Simple-Constraints.html#Simple-Constraints

inline void flush(void *p) {
#ifdef __powerpc__

  /*
  PowerISA_V2.07B p. 773
  dcbf RA,RB,L

  effective address is RA|0 + RB
  this mnemonic has L=0, which is through all cache levels
  write block to storage and mark as invalid in all processors
  */

  /*!

   linux/arch/powerpc/include/asm/cache.h
  */
  asm volatile("dcbf 0, %0"
               : // no outputs
               : "r"(p)
               : "memory");

#elif __amd64__

  /*!

  arch/x86/include/asm/special_insns.h

   p139
  https://www.amd.com/system/files/TechDocs/24594.pdf

  clflush mem8
  */

  asm volatile("clflush %0"
               : "+m"(p)
               : // no inputs
               : // no clobbers
  );
#else
#warning "flush not implemented"
  (void)p;
#endif
}

inline void barrier_all() {

#ifdef __powerpc__

  // sync is a mnemonic for sync 0, heavyweight sync
  asm volatile("sync"
               : // no outputs
               : // no inputs
               : "memory");

#elif __amd64__

  asm volatile("mfence"
               : // no outputs
               : // no inputs
               : "memory");

#else
#warning "barrier_all not implemented"
#endif
}

inline void flush_all(void *p, const size_t n) {
  // cache flush may not be ordered wrt other kinds of accesses
  barrier_all();

  for (size_t i = 0; i < n; i += 32) {
    char *c = static_cast<char *>(p);
    flush(&c[i]);
  }

  // make flushing visible to other accesses
  barrier_all();
}

/*! return the smallest cache line size detected on the platform.
Return 16 if the cache line size could not be detected.
*/
size_t cache_linesize() {
#ifdef __linux__
  long linesize, var;

  var = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
  linesize = var;

  var = sysconf(_SC_LEVEL2_CACHE_LINESIZE);
  linesize = var ? std::min(linesize, var) : linesize;

  var = sysconf(_SC_LEVEL3_CACHE_LINESIZE);
  linesize = var ? std::min(linesize, var) : linesize;

  var = sysconf(_SC_LEVEL4_CACHE_LINESIZE);
  linesize = var ? std::min(linesize, var) : linesize;

  linesize = linesize ? linesize : 16;
  return linesize;
#else
#warning "unsupported platform"
  return 16;
#endif
}