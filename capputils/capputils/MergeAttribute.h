/*
 * MergeAttribute.h
 *
 *  Created on: Jan 24, 2013
 *      Author: tombr
 */

#ifndef CAPPUTILS_ATTRIBUTES_MERGEATTRIBUTE_H_
#define CAPPUTILS_ATTRIBUTES_MERGEATTRIBUTE_H_

#include "IAttribute.h"

#include "ReflectableClass.h"
#include "IClassProperty.h"

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <vector>

namespace capputils {

namespace attributes {

template<class T>
struct merge_trait {
};

template<class T>
struct merge_trait<boost::shared_ptr<std::vector<T> > > {
  typedef boost::shared_ptr<std::vector<T> > collection_t;
  typedef T value_t;

  static void allocate(collection_t& collection) {
    if (!collection)
      collection = boost::make_shared<std::vector<value_t> >();
  }

  static void setValueAt(collection_t& collection, int position, const value_t& value) {
    if ((int)collection->size() <= position)
      collection->resize(position + 1);
    collection->at(position) = value;
  }
};

template<class T>
struct merge_trait<std::vector<T> > {
  typedef std::vector<T> collection_t;
  typedef T value_t;

  static void allocate(collection_t& /*collection*/) { }

  static void setValueAt(collection_t& collection, int position, const value_t& value) {
    if ((int)collection.size() <= position)
      collection.resize(position + 1);
    collection.at(position) = value;
  }
};

class IMergeAttribute : public virtual IAttribute {
public:
  virtual void setValue(capputils::reflection::ReflectableClass& object,
      const capputils::reflection::IClassProperty* property,
      int position,
      const capputils::reflection::ReflectableClass& fromObject,
      const capputils::reflection::IClassProperty* fromProperty) = 0;

  virtual const std::type_info& getValueType() const = 0;
};

template<class T>
class MergeAttribute : public virtual IMergeAttribute {
  typedef T collection_t;
  typedef typename merge_trait<collection_t>::value_t value_t;

public:
  virtual void setValue(capputils::reflection::ReflectableClass& object,
      const capputils::reflection::IClassProperty* property,
      int position,
      const capputils::reflection::ReflectableClass& fromObject,
      const capputils::reflection::IClassProperty* fromProperty)
  {
    using namespace capputils::reflection;

    const ClassProperty<collection_t>* collectionProperty = dynamic_cast<const ClassProperty<collection_t>* >(property);
    const ClassProperty<value_t>* valueProperty = dynamic_cast<const ClassProperty<value_t>* >(fromProperty);

    if (!collectionProperty || !valueProperty)
      return;

    collection_t collection = collectionProperty->getValue(object);
    value_t value = valueProperty->getValue(fromObject);
    merge_trait<collection_t>::allocate(collection);
    merge_trait<collection_t>::setValueAt(collection, position, value);
    collectionProperty->setValue(object, collection);
  }

  virtual const std::type_info& getValueType() const {
    return typeid(value_t);
  }
};

template<class T>
AttributeWrapper* Merge() {
  return new AttributeWrapper(new MergeAttribute<T>());
}

} /* namespace attributes */

} /* namespace capputils */

#endif /* CAPPUTILS_ATTRIBUTES_MERGEATTRIBUTE_H_ */
