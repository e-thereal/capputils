#include "RegisterClass.h"
#include "ReflectableClassFactory.h"

namespace capputils {

namespace reflection {

RegisterClass::RegisterClass(const std::string& classname,
      ConstructorType constructor,
      DestructorType destructor) : classname(classname)
{
  ReflectableClassFactory::getInstance().registerClass(classname, constructor, destructor);
}

RegisterClass::~RegisterClass() {
  ReflectableClassFactory::getInstance().freeClass(classname);
}

}

}
