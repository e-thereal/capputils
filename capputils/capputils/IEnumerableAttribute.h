#pragma once

#ifndef _CAPPUTILS_IENUMERABLEATTRIBUTE_H_
#define _CAPPUTILS_IENUMERABLEATTRIBUTE_H_

#include "IAttribute.h"
#include "IClassProperty.h"
#include "IPropertyIterator.h"

#include <string>

namespace capputils {

namespace reflection {

class ReflectableClass;

}

namespace attributes {

class IEnumerableAttribute: public virtual IAttribute {
public:
  virtual reflection::IPropertyIterator* getPropertyIterator(const reflection::IClassProperty* property) = 0;
};

}

}


#endif