#include "Serializer.h"

#include "SerializeAttribute.h"

using namespace capputils::attributes;

namespace capputils {

void Serializer::writeToFile(const capputils::reflection::ReflectableClass& object, FILE* file) {
  std::vector<capputils::reflection::IClassProperty*>& properties = object.getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    capputils::attributes::ISerializeAttribute* serialize = properties[i]->getAttribute<ISerializeAttribute>();
    if (serialize)
      serialize->writeToFile(properties[i], object, file);
  }
}

bool Serializer::writeToFile(const capputils::reflection::ReflectableClass& object, const std::string& filename) {
  FILE* file = fopen(filename.c_str(), "wb");
  if (!file)
    return false;
    
  writeToFile(object, file);
  fclose(file);

  return true;
}

bool Serializer::readFromFile(capputils::reflection::ReflectableClass& object, FILE* file) {
  std::vector<capputils::reflection::IClassProperty*>& properties = object.getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    capputils::attributes::ISerializeAttribute* serialize = properties[i]->getAttribute<ISerializeAttribute>();
    if (serialize)
      serialize->readFromFile(properties[i], object, file);
  }
  return true;
}

bool Serializer::readFromFile(capputils::reflection::ReflectableClass& object, const std::string& filename) {
  FILE* file = fopen(filename.c_str(), "rb");
  if (!file)
    return false;
    
  readFromFile(object, file);
  fclose(file);

  return true;
}

}