#pragma once

#ifndef _ICLASSPROPERTY_H_
#define _ICLASSPROPERTY_H_

#include <string>
#include <vector>
#include <cstdarg>

#include "IAttribute.h"

namespace capputils {

namespace reflection {

class ReflectableClass;

class IClassProperty {
public:
  virtual const std::vector<attributes::IAttribute*>& getAttributes() const = 0;
  virtual const std::string& getName() const = 0;
  virtual std::string getStringValue(const ReflectableClass& object) const = 0;
  virtual void setStringValue(ReflectableClass& object, const std::string& value) const = 0;
  virtual void* getValuePtr(const ReflectableClass& object) const = 0;
  virtual void setValuePtr(ReflectableClass& object, void* ptr) const = 0;

  template<class AT>
  AT* getAttribute() {
    AT* attribute = 0;
    const std::vector<attributes::IAttribute*>& attributes = getAttributes();
    for (unsigned i = 0; i < attributes.size(); ++i) {
      attribute = dynamic_cast<AT*>(attributes[i]);
      if (attribute != 0)
        return attribute;
    }
    return 0;
  }
};

}

}

#endif
