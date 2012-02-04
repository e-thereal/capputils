#include "SerializeAttribute.h"

namespace capputils {

namespace attributes {

void SerializeAttribute<std::string>::writeToFile(capputils::reflection::ClassProperty<std::string>* prop,
    const capputils::reflection::ReflectableClass& object, std::ostream& file)
{
  assert(prop);

  std::string value = prop->getValue(object);
  size_t count = value.size();
  file.write((char*)&count, sizeof(count));
  file.write((char*)value.c_str(), sizeof(std::string::value_type) * count);
}

void SerializeAttribute<std::string>::readFromFile(capputils::reflection::ClassProperty<std::string>* prop,
    capputils::reflection::ReflectableClass& object, std::istream& file)
{
  assert(prop);
  size_t count = 0;
  file.read((char*)&count, sizeof(count));

  std::string value(count, ' ');
  file.read((char*)&value[0], sizeof(std::string::value_type) * count);
  prop->setValue(object, value);
}

void SerializeAttribute<std::string>::writeToFile(capputils::reflection::IClassProperty* prop,
    const capputils::reflection::ReflectableClass& object, std::ostream& file)
{
  capputils::reflection::ClassProperty<std::string>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<std::string>* >(prop);
  assert(typedProperty);
  writeToFile(typedProperty, object, file);
}

void SerializeAttribute<std::string>::readFromFile(capputils::reflection::IClassProperty* prop,
    capputils::reflection::ReflectableClass& object, std::istream& file)
{
  capputils::reflection::ClassProperty<std::string>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<std::string>* >(prop);
  assert(typedProperty);
  readFromFile(typedProperty, object, file);
}

}

}
