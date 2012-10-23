/*
 * ffttest.cu
 *
 *  Created on: Sep 27, 2012
 *      Author: tombr
 */

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

#include <thrust/sequence.h>

#include <boost/timer.hpp>

typedef tbblas::tensor<float, 2, true> matrix;
typedef tbblas::tensor<tbblas::complex<float>, 2, true> cmatrix;
typedef tbblas::random_tensor<float, 2, true, tbblas::uniform<float> > randu;

void ffttest() {
  using namespace tbblas;

  const int M = 34;
  const int N = 30;

  matrix A = randu(20,15);
  //A[seq(0, 0), seq(2,3)] = ones<float>(2,3);
//  matrix A(5,5);
//  thrust::sequence(A.begin(), A.end());
  cmatrix B = fft(A);

  cmatrix C(A.size());
  real(C) = A;
  cmatrix D = fft(C);

  std::cout << "A = " << A << std::endl;
  std::cout << "fft(A) = " << abs(B) << std::endl;
  std::cout << "fft(A) = " << abs(fftexpand(B)) << std::endl;
  std::cout << "fft(C) = " << abs(D) << std::endl;
  std::cout << "fft(A) = " << fftshift(abs(fftexpand(B))) << std::endl;
  std::cout << "fft(C) = " << fftshift(abs(D)) << std::endl;

#if 1
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
