#pragma once

#ifndef _CAPPUTILS_ENUMERABLEATTRIBUTE_H_
#define _CAPPUTILS_ENUMERABLEATTRIBUTE_H_

#include "IEnumerableAttribute.h"
#include "ClassProperty.h"

namespace capputils {

namespace reflection {

template<class CollectionType, class ValueType>
class PropertyIterator : public virtual IPropertyIterator, public ClassProperty<ValueType> {

public:
  //typedef T CollectionType;
  //typedef typename T::value_type ValueType;

private:
  int i;
  const ClassProperty<CollectionType>* collectionProperty;

public:
  PropertyIterator(const ClassProperty<CollectionType>* collectionProperty)
    : ClassProperty<ValueType>(collectionProperty->getName(), 0, 0, 0), i(0), collectionProperty(collectionProperty)
  {
  }
  virtual ~PropertyIterator() { }

  virtual void reset() {
    i = 0;
  }

  virtual bool eof(const ReflectableClass& object) const {
    return i >= collectionProperty->getValue(object).size();
  }

  virtual void next() {
    ++i;
  }

  virtual ValueType getValue(const ReflectableClass& object) const {
    return collectionProperty->getValue(object)[i];
  }

  virtual void setValue(ReflectableClass& object, const ValueType& value) const {
    CollectionType& collection = collectionProperty->getValue(object);
    if (i < collection.size())
      collection[i] = value;
    else if (i == collection.size())
      collection.push_back(value);
    else
      throw "invalid iterator position!";
    collectionProperty->setValue(object, collection);
  }
};

}

namespace attributes {

template<class T>
class EnumerableAttribute : public virtual IEnumerableAttribute {
public:
  typedef T CollectionType;
  typedef typename T::value_type ValueType;

  virtual reflection::IPropertyIterator* getPropertyIterator(const reflection::IClassProperty* property) {
    using namespace reflection;

    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    if (!typedProperty)
      throw "unexpected error.";
    {
      reflection::PropertyIterator<CollectionType, ValueType> iter(typedProperty);
      iter.reset();
    }
    IPropertyIterator* iter = new reflection::PropertyIterator<CollectionType, ValueType>(typedProperty);
    return iter;
  }
};

template<class T>
AttributeWrapper* Enumerable() {
  return new AttributeWrapper(new EnumerableAttribute<T>());
}

}

}

#endif
