/*
 * ReflectableClass.cpp
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#include "ReflectableClass.h"

namespace reflection {

template<>
std::string convertFromString(const std::string& value) {
  return std::string(value);
}

ReflectableClass::~ReflectableClass() { }

ClassProperty* ReflectableClass::findProperty(const std::string& propertyName) const {
  std::vector<ClassProperty*>& properties = getProperties();

  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i]->name.compare(propertyName) == 0)
      return properties[i];
  }

  return 0;
}

bool ReflectableClass::hasProperty(const std::string& propertyName) const {
  return findProperty(propertyName) != 0;
}

void ReflectableClass::setProperty(const std::string& propertyName, const std::string& propertyValue) {
  ClassProperty* property = findProperty(propertyName);
  if (property)
    property->setValue(*this, propertyValue);
}

const std::string ReflectableClass::getProperty(const std::string& propertyName) {
  ClassProperty* property = findProperty(propertyName);
  if (property)
    return property->getRawValue(*this);

  return "<UNDEFINED>";
}

}
