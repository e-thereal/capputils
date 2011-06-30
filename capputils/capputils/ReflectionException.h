/*
 * ReflectionException.h
 *
 *  Created on: Jun 29, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_REFLECTIONEXCEPTION_H_
#define CAPPUTILS_REFLECTIONEXCEPTION_H_

#include <exception>
#include <string>

namespace capputils {

namespace exceptions {

class ReflectionException : public std::exception {
private:
  std::string cause;
  mutable std::string lastMessage;

public:
  ReflectionException(const std::string& cause);
  virtual ~ReflectionException() throw();
  virtual const char* what() const throw();
};

}

}

#endif /* CAPPUTILS_REFLECTIONEXCEPTION_H_ */
