/*
 * ompsegfault.cu
 *
 *  Created on: Mar 8, 2013
 *      Author: tombr
 */

#include "tests.h"

#include <unistd.h>

void ompsegfault() {

  #pragma omp parallel
  {
    double* ptr;
    cudaMalloc(&ptr, 8);
    sleep(1);
    cudaFree(ptr);
  }
}
