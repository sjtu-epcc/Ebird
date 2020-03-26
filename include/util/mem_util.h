//
// Created by ay27 on 9/5/17.
//

#pragma once

#include <util/common.h>
// #include <cuda_malloc.h>
#include <util/error_util.h>

namespace ebird {

#define BYTE_TO_MB(_size_in_byte) (((double)(_size_in_byte)) / 1024.0 / 1024.0)

inline size_t query_free_mem() {
  size_t mem_tot_0 = 0;
  size_t mem_free_0 = 0;
  cudaMemGetInfo(&mem_free_0, &mem_tot_0);
  return mem_free_0;
}

inline size_t query_used_mem() {
  size_t mem_tot_0 = 0;
  size_t mem_free_0 = 0;
  cudaMemGetInfo(&mem_free_0, &mem_tot_0);
  return mem_tot_0 - mem_free_0;
}

}  // namespace ebird
