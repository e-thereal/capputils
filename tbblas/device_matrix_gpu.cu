/*
 * device_matrix_gpu.cu
 *
 *  Created on: Nov 16, 2011
 *      Author: tombr
 */
#define BOOST_TYPEOF_COMPLIANT

#include "device_matrix.hpp"

const int BlockWidth = 32;
const int BlockHeight = 16;

namespace tbblas {

__global__ void geaxpyKernel(int m, int n, float alpha, const float* A, int lda,
    float* B, int ldb)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= m || j >= n)
    return;

  B[i + j * ldb] += alpha * A[i + j * lda];
}

template<>
void tbblas_geaxpy(int m, int n, float alpha, const float* A, int lda,
    float* B, int ldb)
{
  dim3 gridDim((m + BlockWidth - 1) / BlockWidth,
        (n + BlockHeight - 1) / BlockHeight);
  dim3 blockDim(BlockWidth, BlockHeight);

  geaxpyKernel<<<gridDim, blockDim>>>(m, n, alpha, A, lda, B, ldb);
}

}
