#ifndef _CAPPUTILS_REFLECTABLEATTRIBUTE_H_
#define _CAPPUTILS_REFLECTABLEATTRIBUTE_H_

#include "IReflectableAttribute.h"
#include "ClassProperty.h"

namespace capputils {

namespace attributes {

template<class T>
class ReflectableAttribute : public virtual IReflectableAttribute {
private:
  mutable T value;

public:
  virtual reflection::ReflectableClass* getValuePtr(const reflection::ReflectableClass& object,
      reflection::IClassProperty* property) const
  {
    using namespace capputils::reflection;
    ClassProperty<T>* typedProperty = dynamic_cast<ClassProperty<T>* >(property);
    if (typedProperty) {
      value = typedProperty->getValue(object);
      return &value;
    }
    return 0;
  }

  virtual void setValuePtr(reflection::ReflectableClass& object,
      reflection::IClassProperty* property, reflection::ReflectableClass* valuePtr) const
  {
    using namespace capputils::reflection;
    ClassProperty<T>* typedProperty = dynamic_cast<ClassProperty<T>* >(property);
    if (typedProperty) {
      typedProperty->setValue(object, *((T*)valuePtr));
      delete valuePtr;
    }
  }
};

template<class T>
class ReflectableAttribute<T*> : public virtual IReflectableAttribute {
public:
  virtual reflection::ReflectableClass* getValuePtr(const reflection::ReflectableClass& object,
        reflection::IClassProperty* property) const
    {
      using namespace capputils::reflection;
      ClassProperty<T*>* typedProperty = dynamic_cast<ClassProperty<T*>* >(property);
      if (typedProperty) {
        return typedProperty->getValue(object);
      }
      return 0;
    }

  virtual void setValuePtr(reflection::ReflectableClass& object,
        reflection::IClassProperty* property, reflection::ReflectableClass* valuePtr) const
  {
    using namespace capputils::reflection;
    ClassProperty<T*>* typedProperty = dynamic_cast<ClassProperty<T*>* >(property);
    if (typedProperty) {
      typedProperty->setValue(object, (T*)valuePtr);
    }
  }
};

template<class T>
AttributeWrapper* Reflectable() {
  return new AttributeWrapper(new ReflectableAttribute<T>());
}

}

}

#endif
