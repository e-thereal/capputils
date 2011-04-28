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
class Converter {
public:
  static T fromString(const std::string& value) {
    T result;
    std::stringstream s(value);
    s >> result;
    return result;
  }

  static std::string toString(const T& value) {
    std::stringstream s;
    s << value;
    return s.str();
  }
};

/*** Template specializations for strings (from string variants) ***/

template<class T>
class Converter<std::vector<T> > {
public:
  static std::vector<T> fromString(const std::string& value) {
    std::string result;
    std::stringstream s(value);
    std::vector<T> vec;
    while (!s.eof()) {
      s >> result;
      vec.push_back(Converter<T>::fromString(result));
    }
    return vec;
  }

  static std::string toString(const std::vector<T>& value) {
    std::stringstream s;
    if (value.size())
      s << value[0];
    for (unsigned i = 1; i < value.size(); ++i)
      s << " " << value[i];
    return s.str();
  }
};

template<>
class Converter<std::string> {
public:
  static std::string fromString(const std::string& value) {
    return std::string(value);
  }

  static std::string toString(const std::string& value) {
    return std::string(value);
  }
};

template<class T>
class ClassProperty : public virtual IClassProperty
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
    return Converter<T>::toString(getValue(object));
  }

  virtual void setStringValue(ReflectableClass& object, const std::string& value) const {
    setValue(object, Converter<T>::fromString(value));
  }

  virtual const std::type_info& getType() const {
    return typeid(T);
  }

  virtual T getValue(const ReflectableClass& object) const {
    return getValueFunc(object);
  }

  virtual void setValue(ReflectableClass& object, const T& value) const {
    setValueFunc(object, value);
  }

  virtual void setValue(ReflectableClass& object, const ReflectableClass& fromObject, const IClassProperty* fromProperty) {
    const ClassProperty<T>* typedProperty = dynamic_cast<const ClassProperty<T>*>(fromProperty);
    if (typedProperty)
      setValue(object, typedProperty->getValue(fromObject));
  }
};

template<class T>
class ClassProperty<T*> : public virtual IClassProperty
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
    T* value = getValue(object);
    if (value)
      return Converter<T>::toString(*value);
    else
      return "<null>";
  }

  virtual void setStringValue(ReflectableClass& object, const std::string&/* value*/) const {
    //setValueFunc(object, new T(convertFromString<T>(value)));
    // TODO: static assert that getting pointer values from a string is not supported
    throw "setting pointer values from a string is not supported";
    setValue(object, 0);
  }

  virtual const std::type_info& getType() const {
    return typeid(T*);
  }

  virtual T* getValue(const ReflectableClass& object) const {
    return getValueFunc(object);
  }

  virtual void setValue(ReflectableClass& object, T* value) const {
    setValueFunc(object, value);
  }

  virtual void setValue(ReflectableClass& object, const ReflectableClass& fromObject, const IClassProperty* fromProperty) {
    const ClassProperty<T*>* typedProperty = dynamic_cast<const ClassProperty<T*>*>(fromProperty);
    if (typedProperty)
      setValue(object, typedProperty->getValue(fromObject));
  }
};

}

}

#endif
