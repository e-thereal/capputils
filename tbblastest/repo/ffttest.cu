/*
 * ffttest.cu
 *
 *  Created on: Sep 27, 2012
 *      Author: tombr
 */

//#define TBBLAS_BATCHED_FFT

#include <tbblas/tensor.hpp>
#include <tbblas/io.hpp>
#include <tbblas/random.hpp>
#include <tbblas/complex.hpp>
#include <tbblas/fft.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/math.hpp>
#include <tbblas/expand.hpp>
#include <tbblas/shift.hpp>
#include <tbblas/real.hpp>
#include <tbblas/gaussian.hpp>
#include <tbblas/filter.hpp>

#include <thrust/sequence.h>
#include <tbblas/sequence_iterator.hpp>

#include <boost/timer.hpp>

#include <cstdio>

typedef tbblas::tensor<float, 2, true> matrix;
typedef tbblas::tensor<tbblas::complex<float>, 2, true> cmatrix;
typedef tbblas::random_tensor<float, 2, true, tbblas::uniform<float> > randu;

void ffttest() {
  using namespace tbblas;

  typedef matrix::dim_t dim_t;

//  for (sequence_iterator<dim_t> pos(seq(1,2), seq(3,2)); pos; ++pos) {
//    std::cout << pos.current() << " of " << pos.count() << ": " << *pos << std::endl;
//  }
//
//  tbblas_print(sizeof(uint8_t));

  matrix A = floor(randu(2, 4) * 100);

  tbblas_print(A);
  tbblas_print(A[seq(0,0), seq(1,2), A.size()]);
  tbblas_print(A[seq(0,1), seq(1,2), A.size() - seq(0,1)]);
  tbblas_print(A[seq(0,0), seq(1,2), A.size()] * A[seq(0,1), seq(1,2), A.size() - seq(0,1)]);

//  tbblas_print(A);
//  tbblas_print(fftshift(A));
//  tbblas_print(ifftshift(A));

  return;

  tbblas_print(A[seq(0,0), seq(2,2), A.size()]);
  tbblas_print(A[seq(1,0), seq(2,2), A.size() - seq(1,0)]);
  tbblas_print(A[seq(0,1), seq(2,2), A.size() - seq(0,1)]);
  tbblas_print(A[seq(1,1), seq(2,2), A.size() - seq(1,1)]);

  tbblas_print(A[seq(1,0), seq(2,2), A.size()]);
  tbblas_print(A[seq(0,1), seq(2,2), A.size()]);
  tbblas_print(A[seq(1,1), seq(2,2), A.size()]);

//  matrix B = zeros<float>(A.size());
//  B[seq(0,0), seq(2,2), A.size()] = A[seq(1,1), seq(2,2), A.size()];
//  B[seq(1,1), seq(2,2), A.size()] = ones<float>(A.size() / seq(2,2));

//  tbblas_print(B);


//  const int M = 34;
//  const int N = 30;

  typedef tensor<float, 4, true> tensor_t;
  typedef tensor<complex<float>, 4, true> ctensor_t;
  typedef fft_plan<4> plan_t;

  tensor_t T(40, 52, 32, 96);

  plan_t plan, iplan;
  ctensor_t cT = fft(T, 3, plan);
  T = ifft(cT, 3, iplan);

  for (int i = 0; i < 1000; ++i) {
    cT = fft(T, 3, plan);
    T = ifft(cT, 3, iplan);
  }

//  matrix A = floor(10 * randu(6,6));
//  tbblas_print(A);
//
//  cmatrix cA = fft(A, 1);
//  matrix B = ifft(cA, 1);
//  tbblas_print(B);

//  matrix B = fftshift(A);
//  tbblas_print(B);
//  matrix C = fftshift(A, 1);
//  tbblas_print(C);
//
//  matrix::dim_t size = seq(5,5);
//  matrix::dim_t topleft = (B.size() - size + 1) / 2;
//
//  tbblas_print(B[topleft, size]);

//  {
//    tensor<double, 1, true> A(6), B(6), C, D;
//    A = 1, 2, 3, 4, 5, 6;
//    B = 1, 6, 5, 4, 3, 2;
//    C = filter(A, B);
//    D = filter(A, B, seq(3));
//    tbblas_print(C);
//    tbblas_print(D);
//  }

//  matrix A = randu(20,15);
  //A[seq(0, 0), seq(2,3)] = ones<float>(2,3);
//  matrix A(5,5);
//  thrust::sequence(A.begin(), A.end());
//  cmatrix B = fft(A);

//  cmatrix C(A.size());
//  real(C) = A;
//  cmatrix D = fft(C);


//  std::cout << "A = " << A << std::endl;
//  std::cout << "fft(A) = " << abs(B) << std::endl;
//  std::cout << "fft(A) = " << abs(fftexpand(B)) << std::endl;
//  std::cout << "fft(C) = " << abs(D) << std::endl;
//  std::cout << "fft(A) = " << fftshift(abs(fftexpand(B))) << std::endl;
//  std::cout << "fft(C) = " << fftshift(abs(D)) << std::endl;

#if 0
  {
    typedef float value_t;

    {
      typedef tbblas::tensor<value_t, 1, true> matrix;
      typedef tbblas::tensor<tbblas::complex<value_t>, 1, true> cmatrix;
      typedef tbblas::random_tensor<value_t, 1, true, tbblas::uniform<value_t> > randu;

      matrix A = randu(N);
      cmatrix B = fft(A);
      matrix C = ifft(B);

      std::cout << "Error (float, 1D): " << dot(C - A, C - A) << std::endl;
    }

    {
      typedef tbblas::tensor<value_t, 2, true> matrix;
      typedef tbblas::tensor<tbblas::complex<value_t>, 2, true> cmatrix;
      typedef tbblas::random_tensor<value_t, 2, true, tbblas::uniform<value_t> > randu;

      matrix A = randu(M, N);
      cmatrix B = fft(A);
      matrix C = ifft(B);

      std::cout << "Error (float, 2D): " << dot(C - A, C - A) << std::endl;
    }

    {
      typedef tbblas::tensor<value_t, 3, true> matrix;
      typedef tbblas::tensor<tbblas::complex<value_t>, 3, true> cmatrix;
      typedef tbblas::random_tensor<value_t, 3, true, tbblas::uniform<value_t> > randu;

      matrix A = randu(M, N, 1);
      cmatrix B = fft(A);
      matrix C = ifft(B);

      std::cout << "Error (float, 3D): " << dot(C - A, C - A) << std::endl;
    }
  }

  {
    typedef double value_t;

    {
      typedef tbblas::tensor<value_t, 1, true> matrix;
      typedef tbblas::tensor<tbblas::complex<value_t>, 1, true> cmatrix;
      typedef tbblas::random_tensor<value_t, 1, true, tbblas::uniform<value_t> > randu;

      matrix A = randu(N);
      cmatrix B = fft(A);
      matrix C = ifft(B);

      std::cout << "Error (double, 1D): " << dot(C - A, C - A) << std::endl;
    }

    {
      typedef tbblas::tensor<value_t, 2, true> matrix;
      typedef tbblas::tensor<tbblas::complex<value_t>, 2, true> cmatrix;
      typedef tbblas::random_tensor<value_t, 2, true, tbblas::uniform<value_t> > randu;

      matrix A = randu(M, N);
      cmatrix B = fft(A);
      matrix C = ifft(B);

      std::cout << "Error (double, 2D): " << dot(C - A, C - A) << std::endl;
    }

    {
      typedef tbblas::tensor<value_t, 3, true> matrix;
      typedef tbblas::tensor<tbblas::complex<value_t>, 3, true> cmatrix;
      typedef tbblas::random_tensor<value_t, 3, true, tbblas::uniform<value_t> > randu;

      matrix A = randu(M, N, N);
      cmatrix B = fft(A);
      matrix C = ifft(B);

      std::cout << "Error (double, 3D): " << dot(C - A, C - A) << std::endl;
      cudaThreadSynchronize();
    }

//    {
//      typedef tbblas::tensor<value_t, 4, true> matrix;
//      typedef tbblas::tensor<tbblas::complex<value_t>, 4, true> cmatrix;
//      typedef tbblas::random_tensor<value_t, 4, true, tbblas::uniform<value_t> > randu;
//
//      matrix A = randu(2, 2, 2, 2);
//      cmatrix B = fft(A);
//      matrix C = ifft(B);
//
//      std::cout << "Error (double, 4D): " << dot(C - A, C - A) << std::endl;
//    }
  }
#endif
}
