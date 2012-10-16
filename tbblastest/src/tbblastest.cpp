//============================================================================
// Name        : tbblastest.cpp
// Author      : Tom Brosch
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "tests.h"

#include <iostream>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
  float* mem = 0;
  cudaMalloc((void**)&mem, sizeof(float) * 100);

  helloworld();
  //convtest();
//  sumtest();
//  entropytest();
//  copytest();
  //proxycopy();
  //ffttest();
  //fftbenchmarks();
  //convtest2();
  //scalarexpressions();

  cudaDeviceReset();

  return 0;
}
