/*
 * prod.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: tombr
 */

#include <tbblas/prod.hpp>

#include <cublas.h>
#include <cblas.h>

namespace tbblas {

template<>
void gemm(bool transa, bool transb, int m, int n, int k, float alpha, thrust::device_ptr<const float> A, int lda,
    thrust::device_ptr<const float> B, int ldb, float beta, thrust::device_ptr<float> C, int ldc)
{
  const char ctransa = transa ? 't' : 'n';
  const char ctransb = transb ? 't' : 'n';

  cublasSgemm(ctransa, ctransb, m, n, k, alpha, A.get(), lda, B.get(), ldb, beta, C.get(), ldc);
}

template<>
void gemm(bool transa, bool transb, int m, int n, int k, double alpha, thrust::device_ptr<const double> A, int lda,
    thrust::device_ptr<const double> B, int ldb, double beta, thrust::device_ptr<double> C, int ldc)
{
  const char ctransa = transa ? 't' : 'n';
  const char ctransb = transb ? 't' : 'n';

  cublasDgemm(ctransa, ctransb, m, n, k, alpha, A.get(), lda, B.get(), ldb, beta, C.get(), ldc);
}

template<>
void gemm(bool transa, bool transb, int m, int n, int k, float alpha, const float* A, int lda,
    const float* B, int ldb, float beta, float* C, int ldc)
{
  const CBLAS_TRANSPOSE ctransa = transa ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE ctransb = transb ? CblasTrans : CblasNoTrans;

  cblas_sgemm(CblasColMajor, ctransa, ctransb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
void gemm(bool transa, bool transb, int m, int n, int k, double alpha, const double* A, int lda,
    const double* B, int ldb, double beta, double* C, int ldc)
{
  const CBLAS_TRANSPOSE ctransa = transa ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE ctransb = transb ? CblasTrans : CblasNoTrans;

  cblas_dgemm(CblasColMajor, ctransa, ctransb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

}
