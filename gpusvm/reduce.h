#ifndef GPUSVM_REDUCE_H
#define GPUSVM_REDUCE_H
#include "reductionOperators.h"

namespace gpusvm {

template<int stepSize>
__device__ void sumStep(float* temps) {
  if (threadIdx.x < stepSize) {
    temps[threadIdx.x] += temps[threadIdx.x + stepSize];
  }
  if (stepSize >= 32) {
    __syncthreads();
  }
}

__device__ void sumReduce(float* temps) {
  if (256 < GPUSVM_BLOCKSIZE) sumStep<256>(temps);
  if (128 < GPUSVM_BLOCKSIZE) sumStep<128>(temps);
  if ( 64 < GPUSVM_BLOCKSIZE) sumStep< 64>(temps);
  if ( 32 < GPUSVM_BLOCKSIZE) sumStep< 32>(temps);
  if ( 16 < GPUSVM_BLOCKSIZE) sumStep< 16>(temps);
  if (  8 < GPUSVM_BLOCKSIZE) sumStep<  8>(temps);
  if (  4 < GPUSVM_BLOCKSIZE) sumStep<  4>(temps);
  if (  2 < GPUSVM_BLOCKSIZE) sumStep<  2>(temps);
  if (  1 < GPUSVM_BLOCKSIZE) sumStep<  1>(temps);
}

template<int stepSize>
__device__ void maxStep(float* temps) {
  if (threadIdx.x < stepSize) {
    maxOperator(temps[threadIdx.x], temps[threadIdx.x + stepSize], temps + threadIdx.x);
  }
  if (stepSize >= 32) {
    __syncthreads();
  }
}

__device__ void maxReduce(float* temps) {
  if (256 < GPUSVM_BLOCKSIZE) maxStep<256>(temps);
  if (128 < GPUSVM_BLOCKSIZE) maxStep<128>(temps);
  if ( 64 < GPUSVM_BLOCKSIZE) maxStep< 64>(temps);
  if ( 32 < GPUSVM_BLOCKSIZE) maxStep< 32>(temps);
  if ( 16 < GPUSVM_BLOCKSIZE) maxStep< 16>(temps);
  if (  8 < GPUSVM_BLOCKSIZE) maxStep<  8>(temps);
  if (  4 < GPUSVM_BLOCKSIZE) maxStep<  4>(temps);
  if (  2 < GPUSVM_BLOCKSIZE) maxStep<  2>(temps);
  if (  1 < GPUSVM_BLOCKSIZE) maxStep<  1>(temps);
}

template<int stepSize>
__device__ void argminStep(float* values, int* indices) {
  if (threadIdx.x < stepSize) {
    int compOffset = threadIdx.x + stepSize;
    argMin(indices[threadIdx.x], values[threadIdx.x], indices[compOffset], values[compOffset], indices + threadIdx.x, values + threadIdx.x);
  }
  if (stepSize >= 32) {
    __syncthreads();
  }
}

__device__ void argminReduce(float* values, int* indices) {
  if (256 < GPUSVM_BLOCKSIZE) argminStep<256>(values, indices);
  if (128 < GPUSVM_BLOCKSIZE) argminStep<128>(values, indices);
  if ( 64 < GPUSVM_BLOCKSIZE) argminStep< 64>(values, indices);
  if ( 32 < GPUSVM_BLOCKSIZE) argminStep< 32>(values, indices);
  if ( 16 < GPUSVM_BLOCKSIZE) argminStep< 16>(values, indices);
  if (  8 < GPUSVM_BLOCKSIZE) argminStep<  8>(values, indices);
  if (  4 < GPUSVM_BLOCKSIZE) argminStep<  4>(values, indices);
  if (  2 < GPUSVM_BLOCKSIZE) argminStep<  2>(values, indices);
  if (  1 < GPUSVM_BLOCKSIZE) argminStep<  1>(values, indices);
}

template<int stepSize>
__device__ void argmaxStep(float* values, int* indices) {
  if (threadIdx.x < stepSize) {
    int compOffset = threadIdx.x + stepSize;
    argMax(indices[threadIdx.x], values[threadIdx.x], indices[compOffset], values[compOffset], indices + threadIdx.x, values + threadIdx.x);
  }
  if (stepSize >= 32) {
    __syncthreads();
  }
}

__device__ void argmaxReduce(float* values, int* indices) {
  if (256 < GPUSVM_BLOCKSIZE) argmaxStep<256>(values, indices);
  if (128 < GPUSVM_BLOCKSIZE) argmaxStep<128>(values, indices);
  if ( 64 < GPUSVM_BLOCKSIZE) argmaxStep< 64>(values, indices);
  if ( 32 < GPUSVM_BLOCKSIZE) argmaxStep< 32>(values, indices);
  if ( 16 < GPUSVM_BLOCKSIZE) argmaxStep< 16>(values, indices);
  if (  8 < GPUSVM_BLOCKSIZE) argmaxStep<  8>(values, indices);
  if (  4 < GPUSVM_BLOCKSIZE) argmaxStep<  4>(values, indices);
  if (  2 < GPUSVM_BLOCKSIZE) argmaxStep<  2>(values, indices);
  if (  1 < GPUSVM_BLOCKSIZE) argmaxStep<  1>(values, indices);
}

}

#endif
