//============================================================================
// Name        : tbblastest.cpp
// Author      : Tom Brosch
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "tests.h"

#include <iostream>

int main(int argc, char** argv) {
  helloworld();
  convtest();
//  sumtest();
//  entropytest();
//  copytest();
  proxycopy();
  ffttest();
  fftbenchmarks();
  convtest2();
  scalarexpressions();
  return 0;
}
