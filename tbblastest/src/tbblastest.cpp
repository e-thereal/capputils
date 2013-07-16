//============================================================================
// Name        : tbblastest.cpp
// Author      : Tom Brosch
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "tests.h"

#include <iostream>

class TestClass {
public:
  TestClass() {
    std::cout << "Constructed." << std::endl;
  }

  virtual ~TestClass() {
    std::cout << "Destructed." << std::endl;
  }
};

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

  std::cout << "Start test." << std::endl;

  int i = 3;

  {
    TestClass test[i];
  }

  std::cout << "End test." << std::endl;

  return 0;
}
