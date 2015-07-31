/*
 * proxytests.cu
 *
 *  Created on: Nov 20, 2012
 *      Author: tombr
 */
#include "tests.h"

#include <thrust/sequence.h>

#include <tbblas/tensor.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/io.hpp>
#include <tbblas/indices.hpp>

#include <tbblas/linalg.hpp>

#include <tbblas/deeplearn/serialize_conv_rbm.hpp>

#include <iostream>

typedef tbblas::tensor<float, 4, true> tensor_t;

void func(const tbblas::tensor<float, 2>& A) {

}

void g(tbblas::proxy<tbblas::tensor<float, 2> > p) {
  p = 1;
}

void proxytests() {
  using namespace tbblas;
  using namespace tbblas::deeplearn;

//  conv_rbm_model<float, 4> model;
//  deserialize("model.crbm", model);

  tensor<float, 2> A = indices<float>(3, 3),
      C = A;
  tensor<float, 2, true> B(A);
  B = A;
  tbblas::synchronize();
  func(A);
//  func(B);
  A = indices<float>(3, 3);
  A = 1;
  C = A;

  tbblas_print(B);

  // RW proxies.
  C = indices<float>(1,1);
  A[seq(0,0), seq(1,1)] = C;
  A[seq(0,0), seq(1,1)] = C[seq(0,0), seq(1,1)];
  A[seq(0,0), seq(1,1)] = indices<float>(1,1);
  g(A[seq(0,0), seq(1,1)]);
  g(proxy<tensor<float, 2> >(A));

  const tensor<float, 2> D = C;

  C[seq(0,0), seq(1,1)] = D[seq(0,0), seq(1,1)];
//  D[seq(0,0), seq(1,1)] = C[seq(0,0), seq(1,1)];

#if 0
  tensor<float, 2> A = repeat(indices<float>(4, 1), seq(1, 3));
  tbblas_print(A);

  tensor_t A(160, 208, 54, 10), B = A;
//  tensor_t A(128, 128, 64, 16), B = A;
  tensor_t::dim_t slice_size = B.size(), patch_size = seq(150, 200, 50, 1);
  slice_size[3] = 1;
  tensor_t C(slice_size);

  tbblas::tensor<float, 4, true> fA(A.size());
  tbblas::tensor<int, 4, true> iA(A.size());

  for (size_t i = 0; i < 100; ++i) {
//    A = A + 1;
//    for (int j = 0; j < B.size()[3]; ++j)
//      B[seq(0,0,0,j), patch_size] = B[seq(0,0,0,j), patch_size] + 1;

    A = A * A;
//    for (int j = 0; j < B.size()[3]; ++j)
//      B[seq(0,0,0,j), slice_size] = B[seq(0,0,0,j), slice_size] * C;
    B = B * repeat(C, B.size() / C.size());
  }
#endif
}
