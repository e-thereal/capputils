
#ifndef _CAPPUTILS_IRELFECTABLEATTRIBUTE_H_
#define _CAPPUTILS_IRELFECTABLEATTRIBUTE_H_

#include <capputils/attributes/IAttribute.h>
#include <capputils/reflection/IClassProperty.h>

namespace capputils {

namespace reflection {

class ReflectableClass;

}

namespace attributes {

class IReflectableAttribute : public virtual IAttribute {
public:
  virtual reflection::ReflectableClass* getValuePtr(
      const reflection::ReflectableClass& object,
      const reflection::IClassProperty* property) const = 0;

  virtual void setValuePtr(reflection::ReflectableClass& object,
      reflection::IClassProperty* property, reflection::ReflectableClass* valuePtr) const = 0;

  virtual bool isPointer() const = 0;
  virtual bool isSmartPointer() const = 0;
};

}

}

#endif
