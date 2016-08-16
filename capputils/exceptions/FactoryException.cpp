#include <capputils/exceptions/FactoryException.h>

using namespace std;

namespace capputils {

namespace exceptions {

FactoryException::FactoryException(const string& classname) : classname(classname) { }

FactoryException::~FactoryException() throw() { }

const char* FactoryException::what() const throw() {
  lastMessage = string("Can't create instance of ReflectableClass: ") + classname;
  return lastMessage.c_str();
}

}

}
