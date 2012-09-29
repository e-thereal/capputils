/*
 * ffttest.cu
 *
 *  Created on: Sep 27, 2012
 *      Author: tombr
 */

#include <tbblas/tensor.hpp>
#include <tbblas/io2.hpp>
#include <tbblas/random.hpp>
#include <tbblas/complex.hpp>
#include <tbblas/fft2.hpp>
#include <tbblas/dot2.hpp>

#include <thrust/sequence.h>

#include <boost/timer.hpp>

typedef tbblas::tensor<float, 2, true> matrix;
typedef tbblas::tensor<tbblas::complex<float>, 2, true> cmatrix;
typedef tbblas::random_tensor<float, 2, true, tbblas::uniform<float> > randu;

void ffttest() {
  using namespace tbblas;

  const int N = 64;

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

      matrix A = randu(N, N);
      cmatrix B = fft(A);
      matrix C = ifft(B);

      std::cout << "Error (float, 2D): " << dot(C - A, C - A) << std::endl;
    }

    {
      typedef tbblas::tensor<value_t, 3, true> matrix;
      typedef tbblas::tensor<tbblas::complex<value_t>, 3, true> cmatrix;
      typedef tbblas::random_tensor<value_t, 3, true, tbblas::uniform<value_t> > randu;

      matrix A = randu(N, N, N);
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

      matrix A = randu(N, N);
      cmatrix B = fft(A);
      matrix C = ifft(B);

      std::cout << "Error (double, 2D): " << dot(C - A, C - A) << std::endl;
    }

    {
      typedef tbblas::tensor<value_t, 3, true> matrix;
      typedef tbblas::tensor<tbblas::complex<value_t>, 3, true> cmatrix;
      typedef tbblas::random_tensor<value_t, 3, true, tbblas::uniform<value_t> > randu;

      matrix A = randu(N, N, N);
      cmatrix B = fft(A);
      matrix C = ifft(B);

      std::cout << "Error (double, 3D): " << dot(C - A, C - A) << std::endl;
      cudaThreadSynchronize();

      boost::timer timer;
      for (unsigned i = 0; i < 1000; ++i) {
        matrix D = A - C;
        dot(D, D);
      }
      cudaThreadSynchronize();
      std::cout << "Temporary time (alloc): " << timer.elapsed() << "s." << std::endl;

      matrix D = A - C;
      timer.restart();
      for (unsigned i = 0; i < 1000; ++i) {
        D = A - C;
        dot(D, D);
      }
      cudaThreadSynchronize();
      std::cout << "Temporary time (no alloc): " << timer.elapsed() << "s." << std::endl;

      timer.restart();
      for (unsigned i = 0; i < 1000; ++i)
        dot(C - A, C - A);
      cudaThreadSynchronize();
      std::cout << "Direct time: " << timer.elapsed() << "s." << std::endl;
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
}
