#pragma once

#ifndef _CAPPUTILS_ENUMERABLEATTRIBUTE_H_
#define _CAPPUTILS_ENUMERABLEATTRIBUTE_H_

#include <capputils/attributes/IEnumerableAttribute.h>
#include <capputils/attributes/ReflectableAttribute.h>

#include <capputils/reflection/ClassProperty.h>
#include <vector>
#include <boost/make_shared.hpp>

namespace capputils {

namespace reflection {

template<class T>
struct collection_trait {
  typedef typename T::value_type value_t;

  static size_t size(const T& collection) {
    return collection.size();
  }

  static value_t at(const T& collection, size_t i) {
    return collection.at(i);
  }

  static void setAt(T& collection, size_t i, value_t value) {
    collection.at(i) = value;
  }

  static void push_back(T& collection, const value_t& value) {
    collection.push_back(value);
  }

  static void clear(T& collection) {
    collection.clear();
  }

  static bool valid(const T& /*collection*/) {
    return true;
  }

  static T newInstance() {
    return T();
  }
};

template<class T>
struct collection_trait<T*> {
  typedef typename T::value_type value_t;

  static size_t size(const T* collection) {
    assert(collection);
    return collection->size();
  }

  static value_t at(const T* collection, size_t i) {
    assert(collection);
    return collection->at(i);
  }

  static void setAt(T* collection, size_t i, value_t value) {
    assert(collection);
    collection->at(i) = value;
  }

  static void push_back(T* collection, const value_t& value) {
    assert(collection);
    collection->push_back(value);
  }

  static void clear(T* collection) {
    assert(collection);
    collection->clear();
  }

  static bool valid(const T* collection) {
    return collection;
  }

  static T* newInstance() {
    assert(0); // Possible memory leak.
    return new T();
  }
};

template<class T>
struct collection_trait<boost::shared_ptr<T> > {
  typedef typename T::value_type value_t;

  static size_t size(const boost::shared_ptr<T> collection) {
    assert(collection);
    return collection->size();
  }

  static value_t at(const boost::shared_ptr<T>& collection, size_t i) {
    assert(collection);
    return collection->at(i);
  }

  static void setAt(boost::shared_ptr<T>& collection, size_t i, value_t value) {
    assert(collection);
    collection->at(i) = value;
  }

  static void push_back(boost::shared_ptr<T>& collection, const value_t& value) {
    assert(collection);
    collection->push_back(value);
  }

  static void clear(boost::shared_ptr<T>& collection) {
    assert(collection);
    collection->clear();
  }

  static bool valid(const boost::shared_ptr<T>& collection) {
    return collection.get();
  }

  static boost::shared_ptr<T> newInstance() {
    return boost::make_shared<T>();
  }
};

template<class CollectionType>
class PropertyIterator : public virtual IPropertyIterator,
                         public ClassProperty<typename collection_trait<CollectionType>::value_t>
{

private:
  typedef typename collection_trait<CollectionType>::value_t value_t;

  const ReflectableClass& object;
  const ClassProperty<CollectionType>* collectionProperty;
  int i;

public:
  PropertyIterator(const ReflectableClass& object, const ClassProperty<CollectionType>* collectionProperty)
    : ClassProperty<value_t>(collectionProperty->getName(), value_t(), 0, 0, 0, NULL),
      object(object), collectionProperty(collectionProperty), i(0) { }
  virtual ~PropertyIterator() { }

  virtual void reset() {
    i = 0;
  }

  virtual bool eof() const {
    return i >= (int)collection_trait<CollectionType>::size(collectionProperty->getValue(object));
  }

  virtual void next() {
    ++i;
  }

  virtual void prev() {
    if (i > 0)
      --i;
  }

  virtual void clear(ReflectableClass& object) {
    assert(&this->object == &object);
    CollectionType collection = collectionProperty->getValue(object);
    collection_trait<CollectionType>::clear(collection);
    collectionProperty->setValue(object, collection);
  }

  virtual value_t getValue(const ReflectableClass& object) const {
    assert(&this->object == &object);
    assert(i >= 0);
    assert(i < (int)collection_trait<CollectionType>::size(collectionProperty->getValue(object)));
    return collection_trait<CollectionType>::at(collectionProperty->getValue(object), i);
  }

  virtual void setValue(ReflectableClass& object, value_t value) const {
    assert(&this->object == &object);
    CollectionType collection = collectionProperty->getValue(object);
    assert(i >= 0);
    assert(i <= (int)collection_trait<CollectionType>::size(collection));

    if (i < (int)collection_trait<CollectionType>::size(collection))
      collection_trait<CollectionType>::setAt(collection, i, value);
    else
      collection_trait<CollectionType>::push_back(collection, value);

    collectionProperty->setValue(object, collection);
  }

  virtual void setValue(ReflectableClass& object, const ReflectableClass& fromObject, const IClassProperty* fromProperty) {
    assert(&this->object == &object);
    ClassProperty<value_t>::setValue(object, fromObject, fromProperty);
  }

  virtual int compare(ReflectableClass& object, const ReflectableClass& fromObject, const IClassProperty* fromProperty) {
    const ClassProperty<value_t>* typedProperty = dynamic_cast<const ClassProperty<value_t>*>(fromProperty);
    assert(typedProperty);

    return compare_trait<value_t>::compare(getValue(object), typedProperty->getValue(fromObject));
  }

  virtual void resetValue(ReflectableClass& object) {
    assert(&this->object == &object);
    ClassProperty<value_t>::resetValue(object);
  }
};

}

namespace attributes {

template<class T, bool reflectable = false>
class EnumerableAttribute : public virtual IEnumerableAttribute {
public:
  typedef T CollectionType;

public:
  virtual boost::shared_ptr<reflection::IPropertyIterator> getPropertyIterator(const reflection::ReflectableClass& object, const reflection::IClassProperty* property) {
    using namespace reflection;

    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    assert(typedProperty);
    if (reflection::collection_trait<CollectionType>::valid(typedProperty->getValue(object)))
      return boost::make_shared<PropertyIterator<CollectionType> >(object, typedProperty);
    else
      return boost::shared_ptr<PropertyIterator<CollectionType> >();
  }

  virtual void renewCollection(reflection::ReflectableClass& object, const reflection::IClassProperty* property) {
    using namespace reflection;

    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    assert(typedProperty);

    typedProperty->setValue(object, collection_trait<CollectionType>::newInstance());
  }
};

template<class T>
class EnumerableAttribute<T, true> : public virtual IEnumerableAttribute {
public:
  typedef T CollectionType;

public:
  virtual boost::shared_ptr<reflection::IPropertyIterator> getPropertyIterator(const reflection::ReflectableClass& object, const reflection::IClassProperty* property) {
    using namespace reflection;

    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    assert(typedProperty);

    if (reflection::collection_trait<CollectionType>::valid(typedProperty->getValue(object))) {
      boost::shared_ptr<IPropertyIterator> iter(new reflection::PropertyIterator<CollectionType>(object, typedProperty));
      iter->addAttribute(new ReflectableAttribute<typename reflection::collection_trait<CollectionType>::value_t>());
      return iter;
    } else {
      return boost::shared_ptr<PropertyIterator<CollectionType> >();
    }
  }

  virtual void renewCollection(reflection::ReflectableClass& object, const reflection::IClassProperty* property) {
    using namespace reflection;

    const ClassProperty<CollectionType>* typedProperty = dynamic_cast<const ClassProperty<CollectionType>*>(property);
    assert(typedProperty);

    typedProperty->setValue(object, collection_trait<CollectionType>::newInstance());
  }
};

template<class T, bool reflectable = false>
AttributeWrapper* Enumerable() {
  return new AttributeWrapper(new EnumerableAttribute<T, reflectable>());
}

}

}

#endif
