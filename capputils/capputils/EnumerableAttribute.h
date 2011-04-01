#pragma once

#ifndef _CAPPUTILS_ENUMERABLEATTRIBUTE_H_
#define _CAPPUTILS_ENUMERABLEATTRIBUTE_H_

#include "IEnumerableAttribute.h"
#include "ClassProperty.h"

namespace capputils {

namespace attributes {

template<class T>
class EnumerableAttribute : public virtual IEnumerableAttribute {
public:
  typedef T CollectionType;
  typedef typename T::value_type ValueType;

  virtual std::string getStringItemAt(
      const reflection::ReflectableClass& object,
      const reflection::IClassProperty* property,
      size_t pos) const
  {
    using namespace reflection;
    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    if (typedProperty) {
      return Converter<ValueType>::toString(typedProperty->getValue(object)[pos]);
    }
    return "";
  }

  virtual void addStringItem(reflection::ReflectableClass& object,
      reflection::IClassProperty* property, const std::string& item) const
  {
    using namespace reflection;
    
    ClassProperty<CollectionType>* typedProperty = dynamic_cast<ClassProperty<CollectionType>*>(property);
    if (typedProperty) {
      typedProperty->getValue(object).push_back(Converter<ValueType>::fromString(item));
    }
  }

  virtual size_t getCount(const reflection::ReflectableClass& object,
      const reflection::IClassProperty* property) const
  {
    using namespace reflection;
    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    if (typedProperty) {
      return typedProperty->getValue(object).size();
    }
    return 0;
  }
};

template<class T>
AttributeWrapper* Enumerable() {
  return new AttributeWrapper(new EnumerableAttribute<T>());
}

}

}

#endif
