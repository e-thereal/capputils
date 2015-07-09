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

template<class T>
class Test1 {
protected:
  bool test;
};

template<class T>
class Test2 : Test1<T> {
public:
  void test2() {
    this->test = 1;
  }
};

int main(int argc, char** argv) {

//  randomtest();
//  synctest();
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

//  if (argc != 5) {
//    std::cout << "Usage: " << argv[0] << " <size> <channel count> <filter count> <reps>" << std::endl;
//    return 1;
//  }

//  trainertests(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
//  fasttrainer(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));

//  poolingtest();
  encodertest();
//  swaptest();

  return 0;
}
