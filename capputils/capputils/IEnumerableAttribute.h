#pragma once

#ifndef _CAPPUTILS_IENUMERABLEATTRIBUTE_H_
#define _CAPPUTILS_IENUMERABLEATTRIBUTE_H_

#include "IAttribute.h"
#include "IClassProperty.h"

#include <string>

namespace capputils {

namespace reflection {

class ReflectableClass;

}

namespace attributes {

class IEnumerableAttribute: public virtual IAttribute {
public:
  virtual std::string getStringItemAt(
      const reflection::ReflectableClass& object,
      const reflection::IClassProperty* property,
      size_t pos) const = 0;

  virtual void addStringItem(reflection::ReflectableClass& object,
      reflection::IClassProperty* property, const std::string& item) const = 0;

  virtual size_t getCount(const reflection::ReflectableClass& object,
      const reflection::IClassProperty* property) const = 0;
};

}

}


#endif