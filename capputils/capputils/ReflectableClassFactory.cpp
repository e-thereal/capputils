/*
 * ReflectableClassFactory.cpp
 *
 *  Created on: Mar 8, 2011
 *      Author: tombr
 */

#include "ReflectableClassFactory.h"

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

void ReflectableClassFactory::registerConstructor(
    const string& classname, ConstructorType constructor)
{
  constructors[classname] = constructor;
  classNames.push_back(classname);
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

RegisterConstructor::RegisterConstructor(const std::string& classname,
      ReflectableClassFactory::ConstructorType constructor)
{
  ReflectableClassFactory::getInstance().registerConstructor(classname, constructor);
}

}

}
