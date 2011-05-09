/*
 * ReflectableClassFactory.h
 *
 *  Created on: Mar 8, 2011
 *      Author: tombr
 */

#ifndef REFLECTABLECLASSFACTORY_H_
#define REFLECTABLECLASSFACTORY_H_

#include <map>
#include <string>
#include <vector>

#ifdef _WIN32
#include <boost/interprocess/managed_windows_shared_memory.hpp>
#endif

namespace capputils {

namespace reflection {

class ReflectableClass;

class ReflectableClassFactory {
friend class InitFactory;

public:
  typedef ReflectableClass* (*ConstructorType)();
  typedef void (*DestructorType)(ReflectableClass*);

private:
  std::map<std::string, ConstructorType> constructors;
  std::map<std::string, DestructorType> destructors;
  std::vector<std::string> classNames;
#ifdef _WIN32
  static boost::interprocess::managed_windows_shared_memory* segment;
#endif

public:
  ReflectableClassFactory();
  virtual ~ReflectableClassFactory();

  static ReflectableClassFactory& getInstance();

  ReflectableClass* newInstance(const std::string& classname);
  void deleteInstance(ReflectableClass* instance);
  void registerClass(const std::string& classname, ConstructorType constructor, DestructorType destructor);
  void freeClass(const std::string& classname);
  std::vector<std::string>& getClassNames();
};

class RegisterClass {
private:
  std::string classname;

public:
  RegisterClass(const std::string& classname,
      ReflectableClassFactory::ConstructorType constructor,
      ReflectableClassFactory::DestructorType destructor);
  virtual ~RegisterClass();
};

}

}

#endif /* REFLECTABLECLASSFACTORY_H_ */
