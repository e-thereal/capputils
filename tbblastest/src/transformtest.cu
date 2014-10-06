/*
 * transformtest.cu
 *
 *  Created on: 2014-10-05
 *      Author: tombr
 */

#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/io.hpp>

#include <tbblas/transform/fmatrix4.hpp>
#include <tbblas/transform/io.hpp>
#include <tbblas/transform/transform.hpp>

typedef tbblas::tensor<float, 3, true> volume_t;

void transformtest() {
  using namespace tbblas::transform;

  std::cout << "Transform Test" << std::endl;

  fmatrix4 mat = make_fmatrix4_scaling(0.5, 0.5, 1);
  tbblas_print(mat);

  volume_t vol1(4, 4, 1), vol2;
  vol1 = 0, 1, 2, 0,
         0, 3, 4, 0,
         0, 5, 6, 0,
         0, 7, 8, 0;

  tbblas_print(vol1);

  vol2 = transform(vol1, mat);
  tbblas_print(vol2);
}
