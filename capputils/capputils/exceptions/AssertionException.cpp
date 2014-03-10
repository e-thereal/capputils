/*
 * AssertionException.cpp
 *
 *  Created on: Jun 29, 2011
 *      Author: tombr
 */

#include <capputils/exceptions/AssertionException.h>

using namespace std;

namespace capputils {

namespace exceptions {

AssertionException::AssertionException(const string& cause) : cause(cause) { }

AssertionException::~AssertionException() throw() { }

const char* AssertionException::what() const throw() {
  lastMessage = string("Error in capputils while checking an assertion: ") + cause;
  return lastMessage.c_str();
}

}

}
