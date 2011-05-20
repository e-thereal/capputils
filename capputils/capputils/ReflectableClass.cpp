/*
 * ReflectableClass.cpp
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#include "ReflectableClass.h"

#ifndef _Win32
#include <sstream>
#include <cassert>
#include <iostream>
#endif

using namespace std;

namespace capputils {

namespace reflection {

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

#ifdef _WIN32
std::string trimTypeName(const char* typeName) {
  string name(typeName);
  if (name.find("class") == 0)
    return name.substr(6);
  if (name.find("struct") == 0)
    return name.substr(7);
  return name;
}
#else
std::string trimTypeName(const char* typeName) {
  string name;
  stringstream stream(typeName);
  int num;
  char buffer[256];
  char ch;

  if (typeName[0] == 'N') {
    stream.get(); // read the first N
    for(int i = 0; !stream.eof() && i < 10; ++i) {
      stream >> num; // read number of characters of the namespace name
      assert(num < 256);
      stream.get(buffer, num + 1);
      name += buffer;
      ch = stream.peek();
      if (ch == 'E')
        break;
      else
        name += "::";
    }
  } else {
    stream >> num;
    int cur = stream.tellg();
    name = string(typeName + cur, typeName + cur + num);
  }

  return name;
}
#endif

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

