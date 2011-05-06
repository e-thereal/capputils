/*
 * ReflectableClassFactory.cpp
 *
 *  Created on: Mar 8, 2011
 *      Author: tombr
 */

#include "ReflectableClassFactory.h"
#include "ReflectableClass.h"
#include <iostream>

using namespace std;

namespace capputils {

namespace reflection {

ReflectableClassFactory::ReflectableClassFactory() {
}

ReflectableClassFactory::~ReflectableClassFactory() {
}

ReflectableClass* ReflectableClassFactory::newInstance(const string& classname) {
  if (constructors.find(classname) != constructors.end())
    return constructors[classname]();
  return 0;
}

void ReflectableClassFactory::deleteInstance(ReflectableClass* instance) {
  const std::string& classname = instance->getClassName();
  if (destructors.find(classname) != destructors.end())
    destructors[classname](instance);
}

void ReflectableClassFactory::registerClass(const string& classname,
    ConstructorType constructor, DestructorType destructor)
{
  cout << "Register: " << classname << "(" << this << ")" << endl;
  constructors[classname] = constructor;
  destructors[classname] = destructor;
  classNames.push_back(classname);
}

void ReflectableClassFactory::freeClass(const std::string& classname) {
  cout << "Free constructor: " << classname << "(" << this << ")" << endl;
  constructors.erase(classname);
  destructors.erase(classname);
  for (unsigned i = 0; i < classNames.size(); ++i)
    if (classNames[i].compare(classname) == 0) {
      classNames.erase(classNames.begin() + i);
      break;
    }
}

ReflectableClassFactory& ReflectableClassFactory::getInstance() {
  static ReflectableClassFactory* instance = 0;
  if (!instance)
    instance = new ReflectableClassFactory();
  return *instance;
}

std::vector<std::string>& ReflectableClassFactory::getClassNames() {
  return classNames;
}

RegisterClass::RegisterClass(const std::string& classname,
      ReflectableClassFactory::ConstructorType constructor,
      ReflectableClassFactory::DestructorType destructor) : classname(classname)
{
  ReflectableClassFactory::getInstance().registerClass(classname, constructor, destructor);
}

RegisterClass::~RegisterClass() {
  ReflectableClassFactory::getInstance().freeClass(classname);
}

}

}
