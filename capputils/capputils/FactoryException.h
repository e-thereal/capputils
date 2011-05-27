#pragma once
#ifndef _CAPPUTILS_FACTORYEXCEPTION_H_
#define _CAPPUTILS_FACTORYEXCEPTION_H_

#include <exception>
#include <string>

namespace capputils  {

namespace exceptions {

class FactoryException : public std::exception {
private:
  std::string classname;
  mutable std::string lastMessage;

public:
  FactoryException(const std::string& classname);
  virtual ~FactoryException() throw ();

  virtual const char* what() const throw();
};

}

}

#endif
