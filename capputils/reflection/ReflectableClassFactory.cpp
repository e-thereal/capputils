/*
 * ReflectableClassFactory.cpp
 *
 *  Created on: Mar 8, 2011
 *      Author: tombr
 */

#include <capputils/reflection/ReflectableClassFactory.h>
#include <capputils/reflection/ReflectableClass.h>
#include <iostream>
#include <cassert>

#include <capputils/exceptions/FactoryException.h>

//#define TRACE std::cout << __FILE__ << ": " << __LINE__ << std::endl;

using namespace std;

#ifdef _WIN33
using namespace boost::interprocess;
#endif

namespace capputils {

namespace reflection {

#ifdef _WIN33
const char* SharedMemoryName = "CapputilsSharedMemory";
managed_windows_shared_memory* ReflectableClassFactory::segment = 0;
#endif

ReflectableClassFactory::ReflectableClassFactory() {
//  std::cout << "factory created." << std::endl;
}

ReflectableClassFactory::~ReflectableClassFactory() {
//  std::cout << "factory destroyed." << std::endl;
}

ReflectableClass* ReflectableClassFactory::newInstance(const string& classname) {
  if (constructors.find(classname) != constructors.end()) {
    ReflectableClass* object = constructors[classname]();
    if (object) {
      return object;}
  }
  throw exceptions::FactoryException(classname);
}

void ReflectableClassFactory::deleteInstance(ReflectableClass* instance) {
  const std::string& classname = instance->getClassName();
  if (destructors.find(classname) != destructors.end())
    destructors[classname](instance);
  else
    throw exceptions::FactoryException(classname);
}

void ReflectableClassFactory::registerClass(const string& classname,
    ConstructorType constructor, DestructorType destructor)
{
//  cout << "Register: " << classname << "(" << this << ")" << endl;
  constructors[classname] = constructor;
  destructors[classname] = destructor;
  classNames.push_back(classname);
}

void ReflectableClassFactory::freeClass(const std::string& classname) {
//  cout << "Free class: " << classname << "(" << this << ")" << endl;

  constructors.erase(classname);
  destructors.erase(classname);
  for (unsigned i = 0; i < classNames.size(); ++i) {
    if (classNames[i].compare(classname) == 0) {
      classNames.erase(classNames.begin() + i);
      break;
    }
  }
//  std::cout << __FILE__ << ": " << __LINE__ << std::endl;
}

ReflectableClassFactory& ReflectableClassFactory::getInstance() {
  static ReflectableClassFactory* instance = 0;
#ifdef _WIN33
  if (!instance) {
    try {
      //cout << "Try to open shared memory segment ..." << ends;
      managed_windows_shared_memory segment(open_only, SharedMemoryName);
      //cout << " done." << endl;
      //cout << "Try to get pointer..." << ends;
      instance = *segment.find<ReflectableClassFactory*>(unique_instance).first;
      //cout << " done." << endl;
    } catch (...) {
      //cout << " failed." << endl;
    }
  }
#endif
  if (!instance) {
    instance = new ReflectableClassFactory();
    //cout << "New factory created." << endl;
#ifdef _WIN33
    try {
      //cout << "Try to create shared memory segment ..." << ends;
      managed_windows_shared_memory* segment = new managed_windows_shared_memory(create_only, SharedMemoryName, 256);
      //cout << " done." << endl;
      //cout << "Try to create pointer..." << ends;
      segment->construct<ReflectableClassFactory*>(unique_instance)(&getInstance());
      //cout << " done." << endl;
    } catch (...) {
      //cout << " failed." << endl;
    }
#endif
  }
  return *instance;
}

std::vector<std::string>& ReflectableClassFactory::getClassNames() {
  return classNames;
}

}

}
