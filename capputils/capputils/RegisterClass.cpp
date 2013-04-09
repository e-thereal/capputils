#include "RegisterClass.h"
#include "ReflectableClassFactory.h"

#include <iostream>

namespace capputils {

namespace reflection {

#define TRACE std::cout << __FILE__ << ": " << __LINE__ << std::endl;

RegisterClass::RegisterClass(const std::string& classname,
      ConstructorType constructor,
      DestructorType destructor) : classname(classname)
{
  ReflectableClassFactory::getInstance().registerClass(classname, constructor, destructor);
}

RegisterClass::~RegisterClass() {
//  std::cout << __FILE__ << ": " << __LINE__ << std::endl;
  ReflectableClassFactory::getInstance().freeClass(classname);
//  std::cout << __FILE__ << ": " << __LINE__ << std::endl;
}

}

}
