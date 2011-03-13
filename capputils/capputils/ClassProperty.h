#pragma once

#ifndef _CLASSPROPERTY_H_
#define _CLASSPROPERTY_H_

#include "IClassProperty.h"

#include <sstream>
#include <iostream>

namespace capputils {

namespace reflection {

class ReflectableClass;

template<class T>
T convertFromString(const std::string& value) {
  T result;
  std::stringstream s(value);
  s >> result;
  return result;
}

template<class T>
const std::string convertToString(const T& value) {
  std::stringstream s;
  s << value;
  return s.str();
}

/*** Template specializations for strings ***/

template<>
std::string convertFromString(const std::string& value);

template<class T>
class ClassProperty : public IClassProperty
{
private:
  std::string name;
  std::vector<attributes::IAttribute*> attributes;

  T (*getValueFunc) (const ReflectableClass& object);
  void (*setValueFunc) (ReflectableClass& object, T value);

  mutable T value;

public:
  /* The last parameter is a list of IAttribute* which must be terminated
   * by null.
   */
  ClassProperty(const std::string& name,
      T (*getValue) (const ReflectableClass& object),
      void (*setValue) (ReflectableClass& object, T value),
      ...)
      : name(name), getValueFunc(getValue), setValueFunc(setValue)
  {
    va_list args;
    va_start(args, setValue);

    for (attributes::AttributeWrapper* attrWrap = va_arg(args, attributes::AttributeWrapper*); attrWrap; attrWrap = va_arg(args, attributes::AttributeWrapper*)) {
      attributes.push_back(attrWrap->attribute);
      delete attrWrap;
    }
  }

  virtual const std::vector<attributes::IAttribute*>& getAttributes() const { return attributes; }
  virtual const std::string& getName() const { return name; }

  virtual std::string getStringValue(const ReflectableClass& object) const {
    return convertToString<T>(getValueFunc(object));
  }

  virtual void setStringValue(ReflectableClass& object, const std::string& value) const {
    setValueFunc(object, convertFromString<T>(value));
  }

  virtual const type_info& getType() const {
    return typeid(T);
  }

  T getValue(const ReflectableClass& object) const {
    return getValueFunc(object);
  }

  void setValue(ReflectableClass& object, const T& value) const {
    setValueFunc(object, value);
  }
};

template<class T>
class ClassProperty<T*> : public IClassProperty
{
private:
  std::string name;
  std::vector<attributes::IAttribute*> attributes;

  T* (*getValueFunc) (const ReflectableClass& object);
  void (*setValueFunc) (ReflectableClass& object, T* value);

public:
  /* The last parameter is a list of IAttribute* which must be terminated
   * by null.
   */
  ClassProperty(const std::string& name,
      T* (*getValue) (const ReflectableClass& object),
      void (*setValue) (ReflectableClass& object, T* value),
      ...)
      : name(name), getValueFunc(getValue), setValueFunc(setValue)
  {
    va_list args;
    va_start(args, setValue);

    for (attributes::AttributeWrapper* attrWrap = va_arg(args, attributes::AttributeWrapper*); attrWrap; attrWrap = va_arg(args, attributes::AttributeWrapper*)) {
      attributes.push_back(attrWrap->attribute);
      delete attrWrap;
    }
  }

  virtual const std::vector<attributes::IAttribute*>& getAttributes() const { return attributes; }
  virtual const std::string& getName() const { return name; }

  virtual std::string getStringValue(const ReflectableClass& object) const {
    T* value = getValueFunc(object);
    if (value)
      return convertToString<T>(*value);
    else
      return "<null>";
  }

  virtual void setStringValue(ReflectableClass& object, const std::string&/* value*/) const {
    //setValueFunc(object, new T(convertFromString<T>(value)));
    // TODO: static assert that getting pointer values from a string is not supported
    throw "setting pointer values from a string is not supported";
    setValueFunc(object, 0);
  }

  virtual const type_info& getType() const {
    return typeid(T*);
  }

  T* getValue(const ReflectableClass& object) const {
    return getValueFunc(object);
  }

  void setValue(ReflectableClass& object, T* value) const {
    setValueFunc(object, value);
  }
};

}

}

#endif
