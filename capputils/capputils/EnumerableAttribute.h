#pragma once

#ifndef _CAPPUTILS_ENUMERABLEATTRIBUTE_H_
#define _CAPPUTILS_ENUMERABLEATTRIBUTE_H_

#include "IEnumerableAttribute.h"
#include "ClassProperty.h"
#include "ReflectableAttribute.h"

namespace capputils {

namespace reflection {

template<class CollectionType, class ValueType>
class PropertyIterator : public virtual IPropertyIterator, public ClassProperty<ValueType> {

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
    return i >= (int)collectionProperty->getValue(object).size();
  }

  virtual void next() {
    ++i;
  }

  virtual void prev() {
    if (i > 0)
      --i;
  }

  virtual ValueType getValue(const ReflectableClass& object) const {
    assert(i >= 0);
    assert(i < (int)collectionProperty->getValue(object).size());
    return collectionProperty->getValue(object)[i];
  }

  virtual void setValue(ReflectableClass& object, ValueType value) const {
    CollectionType collection = collectionProperty->getValue(object);
    if (i < 0)
      throw "invalid iterator position! (negative)";
    else if (i < (int)collection.size())
      collection[i] = value;
    else if (i == (int)collection.size())
      collection.push_back(value);
    else
      throw "invalid iterator position!";
    collectionProperty->setValue(object, collection);
  }
};

template<class CollectionType, class ValueType>
class PropertyIterator<boost::shared_ptr<CollectionType>, ValueType> : public virtual IPropertyIterator, public ClassProperty<ValueType> {

private:
  int i;
  const ClassProperty<boost::shared_ptr<CollectionType> >* collectionProperty;

public:
  PropertyIterator(const ClassProperty<boost::shared_ptr<CollectionType> >* collectionProperty)
    : ClassProperty<ValueType>(collectionProperty->getName(), 0, 0, 0), i(0), collectionProperty(collectionProperty)
  {
  }
  virtual ~PropertyIterator() { }

  virtual void reset() {
    i = 0;
  }

  virtual bool eof(const ReflectableClass& object) const {
    return i >= (int)collectionProperty->getValue(object)->size();
  }

  virtual void next() {
    ++i;
  }

  virtual void prev() {
    if (i > 0)
      --i;
  }

  virtual ValueType getValue(const ReflectableClass& object) const {
    assert(i >= 0);
    assert(i < (int)collectionProperty->getValue(object)->size());
    return collectionProperty->getValue(object)->at(i);
  }

  virtual void setValue(ReflectableClass& object, ValueType value) const {
    boost::shared_ptr<CollectionType> collection = collectionProperty->getValue(object);
    if (i < 0)
      throw "invalid iterator position! (negative)";
    else if (i < (int)collection->size())
      (*collection)[i] = value;
    else if (i == (int)collection->size())
      collection->push_back(value);
    else
      throw "invalid iterator position!";
    collectionProperty->setValue(object, collection);
  }
};

template<class CollectionType, class ValueType>
class PropertyIterator<CollectionType*, ValueType> : public virtual IPropertyIterator, public ClassProperty<ValueType> {

private:
  int i;
  const ClassProperty<CollectionType*>* collectionProperty;

public:
  PropertyIterator(const ClassProperty<CollectionType*>* collectionProperty)
    : ClassProperty<ValueType>(collectionProperty->getName(), 0, 0, 0), i(0), collectionProperty(collectionProperty)
  {
  }
  virtual ~PropertyIterator() { }

  virtual void reset() {
    i = 0;
  }

  virtual bool eof(const ReflectableClass& object) const {
    return i >= (int)collectionProperty->getValue(object)->size();
  }

  virtual void next() {
    ++i;
  }

  virtual void prev() {
    if (i > 0)
      --i;
  }

  virtual ValueType getValue(const ReflectableClass& object) const {
    assert(i >= 0);
    assert(i < (int)collectionProperty->getValue(object)->size());
    return collectionProperty->getValue(object)->at(i);
  }

  virtual void setValue(ReflectableClass& object, ValueType value) const {
    CollectionType* collection = collectionProperty->getValue(object);
    if (i < 0)
      throw "invalid iterator position! (negative)";
    else if (i < (int)collection->size())
      (*collection)[i] = value;
    else if (i == (int)collection->size())
      collection->push_back(value);
    else
      throw "invalid iterator position!";
    collectionProperty->setValue(object, collection);
  }
};

}

namespace attributes {

template<class T, bool reflectable>
class EnumerableAttribute : public virtual IEnumerableAttribute {
public:
  typedef T CollectionType;
  typedef typename T::value_type ValueType;

public:
  virtual reflection::IPropertyIterator* getPropertyIterator(const reflection::IClassProperty* property) {
    using namespace reflection;

    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    if (!typedProperty)
      throw "unexpected error.";
    IPropertyIterator* iter = new reflection::PropertyIterator<CollectionType, ValueType>(typedProperty);
    return iter;
  }

  virtual void clear(const reflection::IClassProperty* property, reflection::ReflectableClass& object) {
    using namespace reflection;
    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    typedProperty->setValue(object, CollectionType());
  }
};

template<class T>
class EnumerableAttribute<T, true> : public virtual IEnumerableAttribute {
public:
  typedef T CollectionType;
  typedef typename T::value_type ValueType;

public:
  virtual reflection::IPropertyIterator* getPropertyIterator(const reflection::IClassProperty* property) {
    using namespace reflection;

    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    if (!typedProperty)
      throw "unexpected error.";
    IPropertyIterator* iter = new reflection::PropertyIterator<CollectionType, ValueType>(typedProperty);
    iter->addAttribute(new ReflectableAttribute<ValueType>());
    return iter;
  }

  virtual void clear(const reflection::IClassProperty* property, reflection::ReflectableClass& object) {
    using namespace reflection;
    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    typedProperty->setValue(object, CollectionType());
  }
};

template<class T>
class EnumerableAttribute<boost::shared_ptr<T>, false> : public virtual IEnumerableAttribute {
public:
  typedef boost::shared_ptr<T> CollectionType;
  typedef typename T::value_type ValueType;

  virtual reflection::IPropertyIterator* getPropertyIterator(const reflection::IClassProperty* property) {
    using namespace reflection;

    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    if (!typedProperty)
      throw "unexpected error.";
    IPropertyIterator* iter = new reflection::PropertyIterator<CollectionType, ValueType>(typedProperty);
    return iter;
  }

  virtual void clear(const reflection::IClassProperty* property, reflection::ReflectableClass& object) {
    using namespace reflection;
    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    typedProperty->getValue(object)->clear();
  }
};

template<class T>
class EnumerableAttribute<boost::shared_ptr<T>, true> : public virtual IEnumerableAttribute {
public:
  typedef boost::shared_ptr<T> CollectionType;
  typedef typename T::value_type ValueType;

  virtual reflection::IPropertyIterator* getPropertyIterator(const reflection::IClassProperty* property) {
    using namespace reflection;

    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    if (!typedProperty)
      throw "unexpected error.";
    IPropertyIterator* iter = new reflection::PropertyIterator<CollectionType, ValueType>(typedProperty);
    iter->addAttribute(new ReflectableAttribute<ValueType>());
    return iter;
  }

  virtual void clear(const reflection::IClassProperty* property, reflection::ReflectableClass& object) {
    using namespace reflection;
    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    typedProperty->getValue(object)->clear();
  }
};

template<class T>
class EnumerableAttribute<T*, false> : public virtual IEnumerableAttribute {
public:
  typedef T* CollectionType;
  typedef typename T::value_type ValueType;

  virtual reflection::IPropertyIterator* getPropertyIterator(const reflection::IClassProperty* property) {
    using namespace reflection;

    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    if (!typedProperty)
      throw "unexpected error.";
    IPropertyIterator* iter = new reflection::PropertyIterator<CollectionType, ValueType>(typedProperty);
    return iter;
  }

  virtual void clear(const reflection::IClassProperty* property, reflection::ReflectableClass& object) {
    using namespace reflection;
    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    typedProperty->getValue(object)->clear();
  }
};

template<class T>
class EnumerableAttribute<T*, true> : public virtual IEnumerableAttribute {
public:
  typedef T* CollectionType;
  typedef typename T::value_type ValueType;

  virtual reflection::IPropertyIterator* getPropertyIterator(const reflection::IClassProperty* property) {
    using namespace reflection;

    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    if (!typedProperty)
      throw "unexpected error.";
    IPropertyIterator* iter = new reflection::PropertyIterator<CollectionType, ValueType>(typedProperty);
    iter->addAttribute(new ReflectableAttribute<ValueType>());
    return iter;
  }

  virtual void clear(const reflection::IClassProperty* property, reflection::ReflectableClass& object) {
    using namespace reflection;
    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    typedProperty->getValue(object)->clear();
  }
};

template<class T, bool reflectable>
AttributeWrapper* Enumerable() {
  return new AttributeWrapper(new EnumerableAttribute<T, reflectable>());
}

}

}

#endif
