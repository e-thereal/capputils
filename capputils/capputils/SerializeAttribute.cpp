#include "SerializeAttribute.h"

namespace capputils {

namespace attributes {

void SerializeAttribute<std::string>::writeToFile(capputils::reflection::ClassProperty<std::string>* prop,
    const capputils::reflection::ReflectableClass& object, FILE* file)
{
  assert(prop);

  std::string value = prop->getValue(object);
  size_t count = value.size();
  assert(fwrite(&count, sizeof(size_t), 1, file));
  assert(fwrite(value.c_str(), sizeof(std::string::value_type), count, file) == count);
}

void SerializeAttribute<std::string>::readFromFile(capputils::reflection::ClassProperty<std::string>* prop,
    capputils::reflection::ReflectableClass& object, FILE* file)
{
  assert(prop);
  size_t count = 0;
  assert(fread(&count, sizeof(size_t), 1, file));

  std::string value(count, ' ');
  assert(fread(&value[0], sizeof(std::string::value_type), count, file) == count);
  prop->setValue(object, value);
}

void SerializeAttribute<std::string>::writeToFile(capputils::reflection::IClassProperty* prop,
    const capputils::reflection::ReflectableClass& object, FILE* file)
{
  capputils::reflection::ClassProperty<std::string>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<std::string>* >(prop);
  assert(typedProperty);
  writeToFile(typedProperty, object, file);
}

void SerializeAttribute<std::string>::readFromFile(capputils::reflection::IClassProperty* prop,
    capputils::reflection::ReflectableClass& object, FILE* file)
{
  capputils::reflection::ClassProperty<std::string>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<std::string>* >(prop);
  assert(typedProperty);
  readFromFile(typedProperty, object, file);
}

}

}
