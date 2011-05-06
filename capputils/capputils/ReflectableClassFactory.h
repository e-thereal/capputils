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

namespace capputils {

namespace reflection {

class ReflectableClass;

class ReflectableClassFactory {
public:
  typedef ReflectableClass* (*ConstructorType)();
  typedef void (*DestructorType)(ReflectableClass*);

private:
  std::map<std::string, ConstructorType> constructors;
  std::map<std::string, DestructorType> destructors;
  std::vector<std::string> classNames;

protected:
  ReflectableClassFactory();

public:
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
