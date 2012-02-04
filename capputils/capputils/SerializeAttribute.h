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
class SerializeAttribute : public virtual ISerializeAttribute {
public:
  virtual void writeToFile(capputils::reflection::ClassProperty<T>* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file) {
    assert(prop);
    T value = prop->getValue(object);
    file.write((char*)&value, sizeof(T));
  }

  virtual void readFromFile(capputils::reflection::ClassProperty<T>* prop, capputils::reflection::ReflectableClass& object, std::istream& file) {
    assert(prop);
    T value;
    file.read((char*)&value, sizeof(T));
    prop->setValue(object, value);
  }

  virtual void writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file) {
    capputils::reflection::ClassProperty<T>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<T>* >(prop);
    assert(typedProperty);
    writeToFile(typedProperty, object, file);
  }

  virtual void readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, std::istream& file) {
    capputils::reflection::ClassProperty<T>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<T>* >(prop);
    assert(typedProperty);
    readFromFile(typedProperty, object, file);
  }
};

template<class T>
class SerializeAttribute<boost::shared_ptr<std::vector<T> > > : public virtual ISerializeAttribute {

  typedef boost::shared_ptr<std::vector<T> > PCollectionType;

public:
  virtual void writeToFile(capputils::reflection::ClassProperty<PCollectionType>* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file) {
    assert(prop);
    PCollectionType collection = prop->getValue(object);
    assert(collection);

    unsigned count = collection->size();
    file.write((char*)&count, sizeof(unsigned));
    file.write((char*)&collection->at(0), sizeof(T) * count);
    assert(file.good());
  }

  virtual void readFromFile(capputils::reflection::ClassProperty<PCollectionType>* prop, capputils::reflection::ReflectableClass& object, std::istream& file) {
    assert(prop);
    unsigned count;
    file.read((char*)&count, sizeof(unsigned));

    PCollectionType collection(new std::vector<T>(count));
    file.read((char*)&collection->at(0), sizeof(T) * count);
    prop->setValue(object, collection);
  }

  virtual void writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file) {
    capputils::reflection::ClassProperty<PCollectionType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PCollectionType>* >(prop);
    assert(typedProperty);
    writeToFile(typedProperty, object, file);
  }

  virtual void readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, std::istream& file) {
    capputils::reflection::ClassProperty<PCollectionType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PCollectionType>* >(prop);
    assert(typedProperty);
    readFromFile(typedProperty, object, file);
  }
};

template<>
class SerializeAttribute<std::string> : public virtual ISerializeAttribute {
public:
  virtual void writeToFile(capputils::reflection::ClassProperty<std::string>* prop,
      const capputils::reflection::ReflectableClass& object, std::ostream& file);

  virtual void readFromFile(capputils::reflection::ClassProperty<std::string>* prop,
      capputils::reflection::ReflectableClass& object, std::istream& file);

  virtual void writeToFile(capputils::reflection::IClassProperty* prop,
      const capputils::reflection::ReflectableClass& object, std::ostream& file);

  virtual void readFromFile(capputils::reflection::IClassProperty* prop,
      capputils::reflection::ReflectableClass& object, std::istream& file);
};

template<class T>
AttributeWrapper* Serialize() {
  return new AttributeWrapper(new SerializeAttribute<T>());
}

}

}

#endif /* _CAPPUTILS_ATTRIBUTES_SERIALIZEATTRIBUTE_H_ */
