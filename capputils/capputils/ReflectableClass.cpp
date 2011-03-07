/*
 * ReflectableClass.cpp
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#include "ReflectableClass.h"

namespace capputils {

namespace reflection {

template<>
std::string convertFromString(const std::string& value) {
  return std::string(value);
}

ReflectableClass::~ReflectableClass() { }

IClassProperty* ReflectableClass::findProperty(const std::string& propertyName) const {
  std::vector<IClassProperty*>& properties = getProperties();

  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i]->getName().compare(propertyName) == 0)
      return properties[i];
  }

  return 0;
}

bool ReflectableClass::hasProperty(const std::string& propertyName) const {
  return findProperty(propertyName) != 0;
}

void ReflectableClass::setProperty(const std::string& propertyName, const std::string& propertyValue) {
  IClassProperty* property = findProperty(propertyName);
  if (property)
    property->setStringValue(*this, propertyValue);
}

const std::string ReflectableClass::getProperty(const std::string& propertyName) {
  IClassProperty* property = findProperty(propertyName);
  if (property)
    return property->getStringValue(*this);

  return "<UNDEFINED>";
}

void ReflectableClass::addAttributes(std::vector< ::capputils::attributes::IAttribute*>* attributes, ...) const {
  va_list args;
  va_start(args, attributes);

  for (attributes::AttributeWrapper* attrWrap = va_arg(args, attributes::AttributeWrapper*); attrWrap; attrWrap = va_arg(args, attributes::AttributeWrapper*)) {
    attributes->push_back(attrWrap->attribute);
    delete attrWrap;
  }
}

void ReflectableClass::toStream(std::ostream& stream) const {
  stream << "reflectable";
}

void ReflectableClass::fromStream(std::istream& str) {
}

}

}

std::ostream& operator<< (std::ostream& stream, const capputils::reflection::ReflectableClass& object) {
  object.toStream(stream);
  return stream;
}

std::istream& operator>> (std::istream& stream, capputils::reflection::ReflectableClass& object) {
  object.fromStream(stream);
  return stream;
}
