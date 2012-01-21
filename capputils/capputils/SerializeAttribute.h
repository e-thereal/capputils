#ifndef _CAPPUTILS_ATTRIBUTES_SERIALIZEATTRIBUTE_H_
#define _CAPPUTILS_ATTRIBUTES_SERIALIZEATTRIBUTE_H_

#include "IAttribute.h"

#include "ReflectableClass.h"
#include "ClassProperty.h"

#include <cstdio>
#include <cassert>

namespace capputils {

namespace attributes {

class ISerializeAttribute : public IAttribute {
public:
  virtual ~ISerializeAttribute() { }

  virtual void writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file) = 0;
  virtual void readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file) = 0;
};

template<class T>
class SerializeAttribute : public virtual ISerializeAttribute {
public:
  virtual void writeToFile(capputils::reflection::ClassProperty<T>* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
    assert(prop);
    T value = prop->getValue(object);
    assert(fwrite(&value, sizeof(T), 1, file) == 1);
  }

  virtual void readFromFile(capputils::reflection::ClassProperty<T>* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
    assert(prop);
    T value;
    assert(fread(&value, sizeof(T), 1, file) == 1);
    prop->setValue(object, value);
  }

  virtual void writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
    capputils::reflection::ClassProperty<T>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<T>* >(prop);
    assert(typedProperty);
    writeToFile(typedProperty, object, file);
  }

  virtual void readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
    capputils::reflection::ClassProperty<T>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<T>* >(prop);
    assert(typedProperty);
    readFromFile(typedProperty, object, file);
  }

};

template<class T>
class SerializeAttribute<boost::shared_ptr<std::vector<T> > > : public virtual ISerializeAttribute {

  typedef boost::shared_ptr<std::vector<T> > PCollectionType;

public:
  virtual void writeToFile(capputils::reflection::ClassProperty<PCollectionType>* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
    assert(prop);
    PCollectionType collection = prop->getValue(object);
    assert(collection);

    unsigned count = collection->size();
    assert(fwrite(&count, sizeof(unsigned), 1, file) == 1);
    assert(fwrite(&collection->at(0), sizeof(T), count, file) == count);
  }

  virtual void readFromFile(capputils::reflection::ClassProperty<PCollectionType>* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
    assert(prop);
    unsigned count;
    assert(fread(&count, sizeof(unsigned), 1, file) == 1);

    PCollectionType collection(new std::vector<T>(count));
    assert(fread(&collection->at(0), sizeof(T), count, file) == count);
    prop->setValue(object, collection);
  }

  virtual void writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
    capputils::reflection::ClassProperty<PCollectionType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PCollectionType>* >(prop);
    assert(typedProperty);
    writeToFile(typedProperty, object, file);
  }

  virtual void readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
    capputils::reflection::ClassProperty<PCollectionType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PCollectionType>* >(prop);
    assert(typedProperty);
    readFromFile(typedProperty, object, file);
  }
};

template<class T>
AttributeWrapper* Serialize() {
  return new AttributeWrapper(new SerializeAttribute<T>());
}

}

}

#endif /* _CAPPUTILS_ATTRIBUTES_SERIALIZEATTRIBUTE_H_ */
