/*
 * device_matrix.cpp
 *
 *  Created on: Nov 16, 2011
 *      Author: tombr
 */

#include "device_matrix.hpp"

#include <iostream>

namespace tbblas {

template<>
void gemm(char transa, char transb, int m, int n, int k, float alpha, const float* A, int lda,
    const float* B, int ldb, float beta, float *C, int ldc)
{
  cublasSgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
void gemm(char transa, char transb, int m, int n, int k, double alpha, const double* A, int lda,
    const double* B, int ldb, double beta, double *C, int ldc)
{
  cublasDgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

}

