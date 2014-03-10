#pragma once
#ifndef CAPPUTILS_REGISTERCLASS_H_
#define CAPPUTILS_REGISTERCLASS_H_

#include <string>

namespace capputils {

namespace reflection {

class ReflectableClass;

typedef ReflectableClass* (*ConstructorType)();
typedef void (*DestructorType)(ReflectableClass*);

class RegisterClass {
private:
  std::string classname;

public:
  RegisterClass(const std::string& classname,
      ConstructorType constructor,
      DestructorType destructor);
  virtual ~RegisterClass();
};

}

}

#endif /* CAPPUTILS_REGISTERCLASS_H_ */