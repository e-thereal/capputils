/*
 * ReflectionException.cpp
 *
 *  Created on: Jun 29, 2011
 *      Author: tombr
 */

#include "ReflectionException.h"

using namespace std;

namespace capputils {

namespace exceptions {

ReflectionException::ReflectionException(const string& cause) : cause(cause) { }

ReflectionException::~ReflectionException() throw() { }

const char* ReflectionException::what() const throw() {
  lastMessage = string("Error in capputils: ") + cause;
  return lastMessage.c_str();
}

}

}
