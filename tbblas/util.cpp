/*
 * util.cpp
 *
 *  Created on: Apr 7, 2014
 *      Author: tombr
 */

#include <tbblas/util.hpp>

#include <cuda_runtime.h>
#include <tbblas/context.hpp>
#include <tbblas/assert.hpp>

namespace tbblas {

int peer_access_enabled_count = 0;

void enable_peer_access(int gpu_count) {
  ++peer_access_enabled_count;

  // If it was already enabled, don't enable it again
  if (peer_access_enabled_count > 1)
    return;

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  tbblas_assert(gpu_count > 0);
  tbblas_assert(gpu_count <= deviceCount);

  for (int tid = 0; tid < gpu_count; ++tid) {
    cudaSetDevice(tid);

    // Enable peer to peer access of each card with the master card and vice versa
    if (tid == 0) {
      for (int i = 1; i < gpu_count; ++i)
        cudaDeviceEnablePeerAccess(i, 0);
    } else {
      cudaDeviceEnablePeerAccess(0, 0);
    }
  }
  cudaSetDevice(0);
}

void disable_peer_access(int gpu_count) {
  if (--peer_access_enabled_count > 0)
    return;

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  tbblas_assert(gpu_count > 0);
  tbblas_assert(gpu_count <= deviceCount);


  for (int tid = 0; tid < gpu_count; ++tid) {
    cudaSetDevice(tid);

    // Enable peer to peer access of each card with the master card and vice versa
    if (tid == 0) {
      for (int i = 1; i < gpu_count; ++i)
        cudaDeviceDisablePeerAccess(i);
    } else {
      cudaDeviceDisablePeerAccess(0);
    }
  }
  cudaSetDevice(0);
}

void synchronize() {
  cudaStreamSynchronize(tbblas::context::get().stream);
}

}
