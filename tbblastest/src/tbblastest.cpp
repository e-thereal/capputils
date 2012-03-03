//============================================================================
// Name        : tbblastest.cpp
// Author      : Tom Brosch
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "tests.h"

#include <iostream>

template<class T, int dim>
class Test_base {
public:
  virtual void printDim() {
    std::cout << "General dim" << std::endl;
  }

  virtual void printType() {
    std::cout << "General type" << std::endl;
  }

  virtual void print() {
    printDim();
    printType();
  }
};

template<class T, class Base>
class TypeHook : public Base {
};

template<int dim, class Base>
class DimHook : public Base {
};

template<class T, int dim>
class Test : public TypeHook<T, DimHook<dim, Test_base<T, dim> > > {
};

template<class Base>
class TypeHook<int, Base> : public Base {
public:
  virtual void printType() {
    std::cout << "int case" << std::endl;
  }
};

template<class Base>
class DimHook<3, Base> : public Base {
public:
  virtual void printDim() {
    std::cout << "3D case" << std::endl;
  }
};

int main() {
  return runtests();
}
