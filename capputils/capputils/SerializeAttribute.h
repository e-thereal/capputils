#ifndef _CAPPUTILS_ATTRIBUTES_SERIALIZEATTRIBUTE_H_
#define _CAPPUTILS_ATTRIBUTES_SERIALIZEATTRIBUTE_H_

#include "IAttribute.h"

#include "ReflectableClass.h"
#include "ClassProperty.h"

#include <cstdio>

namespace capputils {

namespace attributes {

class ISerializeAttribute : public IAttribute {
public:
  virtual ~ISerializeAttribute() { }

  virtual bool writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file) = 0;
  virtual bool readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file) = 0;
};

template<class T>
class SerializeAttribute : public virtual ISerializeAttribute {
public:
  virtual bool writeToFile(capputils::reflection::ClassProperty<T>* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
    assert(prop);
    T value = prop->getValue(object);
    return fwrite(&value, sizeof(T), 1, file) == 1;
  }

  virtual bool readFromFile(capputils::reflection::ClassProperty<T>* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
    assert(prop);
    T value;
    fread(&value, sizeof(T), 1, file);
    prop->setValue(object, value);

    return true;
  }

  virtual bool writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
    capputils::reflection::ClassProperty<T>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<T>* >(prop);
    if (typedProperty)
      return writeToFile(typedProperty, object, file);
    return false;
  }

  virtual bool readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
    capputils::reflection::ClassProperty<T>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<T>* >(prop);
    if (typedProperty)
      return readFromFile(typedProperty, object, file);
    return false;
  }

};

template<class T>
class SerializeAttribute<boost::shared_ptr<std::vector<T> > > : public virtual ISerializeAttribute {

  typedef boost::shared_ptr<std::vector<T> > PCollectionType;

public:
  virtual bool writeToFile(capputils::reflection::ClassProperty<PCollectionType>* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
    assert(prop);
    PCollectionType collection = prop->getValue(object);
    assert(collection);

    unsigned count = collection->size();
    fwrite(&count, sizeof(unsigned), 1, file);
    fwrite(&collection->at(0), sizeof(T), count, file);

    return true;
  }

  virtual bool readFromFile(capputils::reflection::ClassProperty<PCollectionType>* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
    assert(prop);
    unsigned count;
    fread(&count, sizeof(unsigned), 1, file);

    PCollectionType collection(new std::vector<T>(count));
    fread(&collection->at(0), sizeof(T), count, file);

    prop->setValue(object, collection);

    return true;
  }

  virtual bool writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
    capputils::reflection::ClassProperty<PCollectionType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PCollectionType>* >(prop);
    if (typedProperty)
      return writeToFile(typedProperty, object, file);
    return false;
  }

  virtual bool readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
    capputils::reflection::ClassProperty<PCollectionType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PCollectionType>* >(prop);
    if (typedProperty)
      return readFromFile(typedProperty, object, file);
    return false;
  }
};

template<class T>
AttributeWrapper* Serialize() {
  return new AttributeWrapper(new SerializeAttribute<T>());
}

}

}

#endif /* _CAPPUTILS_ATTRIBUTES_SERIALIZEATTRIBUTE_H_ */