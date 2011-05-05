/*
 * ReflectableClassFactory.cpp
 *
 *  Created on: Mar 8, 2011
 *      Author: tombr
 */

#include "ReflectableClassFactory.h"
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

void ReflectableClassFactory::registerConstructor(
    const string& classname, ConstructorType constructor)
{
  cout << "Register: " << classname << "(" << this << ")" << endl;
  constructors[classname] = constructor;
  classNames.push_back(classname);
}

void ReflectableClassFactory::freeConstructor(const std::string& classname) {
  cout << "Free constructor: " << classname << "(" << this << ")" << endl;
  constructors.erase(classname);
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

RegisterConstructor::RegisterConstructor(const std::string& classname,
      ReflectableClassFactory::ConstructorType constructor) : classname(classname)
{
  ReflectableClassFactory::getInstance().registerConstructor(classname, constructor);
}

RegisterConstructor::~RegisterConstructor() {
  ReflectableClassFactory::getInstance().freeConstructor(classname);
}

}

}
