#include "device_vector.hpp"

namespace tbblas {

template<>
void axpy(int n, const float alpha, const float* x, int incx, float* y, int incy) {
  cublasSaxpy(n, alpha, x, incx, y, incy);
}

template<>
void axpy(int n, const double alpha, const double* x, int incx, double* y, int incy) {
  cublasDaxpy(n, alpha, x, incx, y, incy);
}

template<>
float asum(int n, const float* x, int incx) {
  return cublasSasum(n, x, incx);
}

template<>
double asum(int n, const double* x, int incx) {
  return cublasDasum(n, x, incx);
}

template<>
float nrm2(int n, const float* x, int incx) {
  return cublasSnrm2(n, x, incx);
}

template<>
double nrm2(int n, const double* x, int incx) {
  return cublasDnrm2(n, x, incx);
}

template<>
void swap(int n, float* x, int incx, float* y, int incy) {
  cublasSswap(n, x, incx, y, incy);
}

template<>
void swap(int n, double* x, int incx, double* y, int incy) {
  cublasDswap(n, x, incx, y, incy);
}

}
