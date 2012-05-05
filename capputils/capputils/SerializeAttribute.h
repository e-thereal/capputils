#ifndef _CAPPUTILS_ATTRIBUTES_SERIALIZEATTRIBUTE_H_
#define _CAPPUTILS_ATTRIBUTES_SERIALIZEATTRIBUTE_H_

#include "IAttribute.h"

#include "ReflectableClass.h"
#include "ClassProperty.h"

#include <iostream>
#include <cassert>

namespace capputils {

namespace attributes {

class ISerializeAttribute : public IAttribute {
public:
  virtual ~ISerializeAttribute() { }

  virtual void writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file) = 0;
  virtual void readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, std::istream& file) = 0;
};

template<class T>
class serialize_trait {
public:
  static void writeToFile(const T& value, std::ostream& file) {
    file.write((char*)&value, sizeof(T));
  }

  static void readFromFile(T& value, std::istream& file) {
    file.read((char*)&value, sizeof(T));
  }
};

template<class T>
class SerializeAttribute : public virtual ISerializeAttribute {
public:
  virtual void writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file) {
    capputils::reflection::ClassProperty<T>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<T>* >(prop);
    assert(typedProperty);
    serialize_trait<T>::writeToFile(typedProperty->getValue(object), file);
  }

  virtual void readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, std::istream& file) {
    capputils::reflection::ClassProperty<T>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<T>* >(prop);
    assert(typedProperty);
    T value;
    serialize_trait<T>::readFromFile(value, file);
    typedProperty->setValue(object, value);
  }
};

template<class T>
class serialize_trait<boost::shared_ptr<std::vector<T> > >  {
  typedef boost::shared_ptr<std::vector<T> > PCollectionType;
  typedef T value_t;

public:
  static void writeToFile(const PCollectionType& collection, std::ostream& file) {
    unsigned count = collection->size();
    file.write((char*)&count, sizeof(unsigned));
    for (unsigned i = 0; i < count; ++i)
       serialize_trait<value_t>::writeToFile(collection->at(i), file);
    assert(file.good());
  }

  static void readFromFile(PCollectionType& collection, std::istream& file) {
    unsigned count;
    file.read((char*)&count, sizeof(unsigned));

    collection = boost::shared_ptr<std::vector<T> >(new std::vector<T>());
    for (unsigned i = 0; i < count; ++i) {
      value_t value;
      serialize_trait<value_t>::readFromFile(value, file);
      collection->push_back(value);
    }
  }
};

template<>
class serialize_trait<std::string> {
public:
  static void writeToFile(const std::string& value, std::ostream& file);
  static void readFromFile(std::string& value, std::istream& file);
};

template<class T>
AttributeWrapper* Serialize() {
  return new AttributeWrapper(new SerializeAttribute<T>());
}

}

}

#endif /* _CAPPUTILS_ATTRIBUTES_SERIALIZEATTRIBUTE_H_ */
