#pragma once
#ifndef _CAPPUTILS_FACTORYEXCEPTION_H_
#define _CAPPUTILS_FACTORYEXCEPTION_H_

#include <exception>

namespace capputils  {

namespace exceptions {

class FactoryException : public std::exception {

public:
  virtual const char* what() const throw() {
    return "Can't create instance of ReflectableClass";
  }
};

}

}

#endif