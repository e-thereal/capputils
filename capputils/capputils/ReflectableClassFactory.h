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
  typedef ReflectableClass* (*ConstructorType)() ;

private:
  std::map<std::string, ConstructorType> constructors;
  std::vector<std::string> classNames;

protected:
  ReflectableClassFactory();

public:
  virtual ~ReflectableClassFactory();

  static ReflectableClassFactory& getInstance();

  ReflectableClass* newInstance(const std::string& classname);
  void registerConstructor(const std::string& classname, ConstructorType constructor);
  void freeConstructor(const std::string& classname);
  std::vector<std::string>& getClassNames();
};

class RegisterConstructor {
private:
  std::string classname;

public:
  RegisterConstructor(const std::string& classname,
      ReflectableClassFactory::ConstructorType constructor);
  virtual ~RegisterConstructor();
};

}

}

#endif /* REFLECTABLECLASSFACTORY_H_ */
