#pragma once
#ifndef CAPPUTILS_PROPERTYMAP_H_
#define CAPPUTILS_PROPERTYMAP_H_

#include <capputils/capputils.h>

#include <map>
#include <string>

#include <capputils/Variant.h>
#include <capputils/exceptions/ReflectionException.h>

namespace capputils {

namespace reflection {
class ReflectableClass;
}

class PropertyMap {
private:
  typedef std::map<std::string, IVariant*> map_t;
  map_t values;

public:
  PropertyMap(const reflection::ReflectableClass& object);
  virtual ~PropertyMap();

  template<class T>
  bool tryGetValue(const std::string& name, T& value) {
    map_t::iterator pos = values.find(name);
    if (pos != values.end()) {
      Variant<T>* typedValue = dynamic_cast<Variant<T>*>(pos->second);
      if (typedValue) {
        value = typedValue->getValue();
        return true;
      }
    }
    return false;
  }

  template<class T>
  T getValue(const std::string& name) {
    T value;
    if (tryGetValue<T>(name, value))
      return value;
    throw exceptions::ReflectionException("Cannot cast variant to specified type");
  }

  template<class T>
  bool setValue(const std::string& name, const T& value) {
    map_t::iterator pos = values.find(name);
    if (pos != values.end()) {
      Variant<T>* typedValue = dynamic_cast<Variant<T>*>(pos->second);
      if (typedValue) {
        typedValue->setValue(value);
        return true;
      }
    }
    return false;
  }

  void writeToReflectableClass(reflection::ReflectableClass& object);
};

} /* namespace capputils */

#endif /* CAPPUTILS_PROPERTYMAP_H_ */

