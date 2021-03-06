/*
 * prod.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: tombr
 */

#include <tbblas/prod.hpp>
#include <tbblas/context.hpp>

#ifdef TBBLAS_HAVE_CUBLAS
#include <cublas_v2.h>
#endif

#ifdef TBBLAS_HAVE_CBLAS
#include <cblas.h>
#endif

namespace tbblas {

#ifdef TBBLAS_HAVE_CUBLAS
template<>
void gemm(bool transa, bool transb, int m, int n, int k, float alpha, thrust::device_ptr<const float> A, int lda,
    thrust::device_ptr<const float> B, int ldb, float beta, thrust::device_ptr<float> C, int ldc)
{
  const cublasOperation_t ctransa = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t ctransb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasSetStream(context::get().cublasHandle, context::get().stream);
  cublasSgemm(context::get().cublasHandle, ctransa, ctransb, m, n, k, &alpha, A.get(), lda, B.get(), ldb, &beta, C.get(), ldc);
}

template<>
void gemm(bool transa, bool transb, int m, int n, int k, double alpha, thrust::device_ptr<const double> A, int lda,
    thrust::device_ptr<const double> B, int ldb, double beta, thrust::device_ptr<double> C, int ldc)
{
  const cublasOperation_t ctransa = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t ctransb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasSetStream(context::get().cublasHandle, context::get().stream);
  cublasDgemm(context::get().cublasHandle, ctransa, ctransb, m, n, k, &alpha, A.get(), lda, B.get(), ldb, &beta, C.get(), ldc);
}
#endif

#ifdef TBBLAS_HAVE_CBLAS
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
#endif

}
