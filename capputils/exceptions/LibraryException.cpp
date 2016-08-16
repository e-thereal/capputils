#include <capputils/exceptions/LibraryException.h>

using namespace std;

namespace capputils {

namespace exceptions {

LibraryException::LibraryException(const string& library, const string& cause)
 : library(library), cause(cause)
{

}

LibraryException::~LibraryException() throw() { }

const char* LibraryException::what() const throw() {
  lastMessage = string("Can't load library \"") + library + "\" for the following reason:\n" + cause;
  return lastMessage.c_str();
}

}

}
