#pragma once
#ifndef _CAPPUTILS_LIBRARYEXCEPTION_H_
#define _CAPPUTILS_LIBRARYEXCEPTION_H_

#include <exception>

namespace capputils {

namespace exceptions {

class LibraryException : public std::exception {
public:
  virtual const char* what() const throw() {
    return "Can't load library.";
  }
};

}

}

#endif