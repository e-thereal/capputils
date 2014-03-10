/*
 * AssertionException.h
 *
 *  Created on: Jun 29, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_ASSERTIONEXCEPTION_H_
#define CAPPUTILS_ASSERTIONEXCEPTION_H_

#include <exception>
#include <string>

namespace capputils {

namespace exceptions {

class AssertionException : public std::exception {
private:
  std::string cause;
  mutable std::string lastMessage;

public:
  AssertionException(const std::string& cause);
  virtual ~AssertionException() throw();
  virtual const char* what() const throw();
};

}

}

#endif /* CAPPUTILS_ASSERTIONEXCEPTION_H_ */
