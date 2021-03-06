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

#ifdef _WIN33
#include <boost/interprocess/managed_windows_shared_memory.hpp>
#endif

#include <capputils/reflection/RegisterClass.h>

namespace capputils {

namespace reflection {

class ReflectableClass;

class ReflectableClassFactory {
friend class InitFactory;

private:
  std::map<std::string, ConstructorType> constructors;
  std::map<std::string, DestructorType> destructors;
  std::vector<std::string> classNames;
#ifdef _WIN33
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



}

}

#endif /* REFLECTABLECLASSFACTORY_H_ */
