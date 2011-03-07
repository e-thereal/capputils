
#ifndef _CAPPUTILS_IRELFECTABLEATTRIBUTE_H_
#define _CAPPUTILS_IRELFECTABLEATTRIBUTE_H_

#include "IAttribute.h"

namespace capputils {

namespace reflection {

class ReflectableClass;

}

namespace attributes {

class IReflectableAttribute : public virtual IAttribute {
public:
  virtual reflection::ReflectableClass* createInstance() const = 0;
};

}

}

#endif