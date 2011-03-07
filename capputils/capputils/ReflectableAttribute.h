#ifndef _CAPPUTILS_REFLECTABLEATTRIBUTE_H_
#define _CAPPUTILS_REFLECTABLEATTRIBUTE_H_

#include "IReflectableAttribute.h"

namespace capputils {

namespace attributes {

template<class T>
class ReflectableAttribute : public virtual IReflectableAttribute {
public:
  virtual reflection::ReflectableClass* createInstance() const {
    return new T();
  }
};

template<class T>
class ReflectableAttribute<T*> : public virtual IReflectableAttribute {
public:
  virtual reflection::ReflectableClass* createInstance() const {
    return new T();
  }
};

template<class T>
AttributeWrapper* Reflectable() {
  return new AttributeWrapper(new ReflectableAttribute<T>());
}

}

}

#endif