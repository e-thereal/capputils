//============================================================================
// Name        : tbblastest.cpp
// Author      : Tom Brosch
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "tests.h"

#include <iostream>
#include <cstdlib>

int i = 1;

int main(int argc, char** argv) {

//  ompsegfault();
//  helloworld();
//  convtest();
//  sumtest();
//  entropytest();
//  proxycopy();
//  ffttest();
//  fftbenchmarks();
//  convtest2();
//  scalarexpressions();
//  proxytests();
//  convrbmtests();
//  partialffttest();
//  fftflip();
//  maskstest();
//  multigpu();
//  copytest();
//  benchmarks();
//  rearrangetest();

  if (argc != 5) {
    std::cout << "Usage: " << argv[0] << " <filterCount> <channelCount> <reps> <convnetreps>" << std::endl;
    return 1;
  }

  trainertests(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));

  return 0;
}
