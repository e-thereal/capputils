#pragma once
#ifndef _CAPPUTILS_LIBRARYEXCEPTION_H_
#define _CAPPUTILS_LIBRARYEXCEPTION_H_

#include <exception>
#include <string>

namespace capputils {

namespace exceptions {

class LibraryException : public std::exception {
private:
  std::string library;
  std::string cause;
  mutable std::string lastMessage;

public:
  LibraryException(const std::string& library, const std::string& cause);
  virtual ~LibraryException() throw();
  virtual const char* what() const throw();
};

}

}

#endif
