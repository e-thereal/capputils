#include "Serializer.h"

#include "SerializeAttribute.h"
#include <fstream>

using namespace capputils::attributes;

namespace capputils {

void Serializer::writeToFile(const capputils::reflection::ReflectableClass& object, std::ostream& file) {
  std::vector<capputils::reflection::IClassProperty*>& properties = object.getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    capputils::attributes::ISerializeAttribute* serialize = properties[i]->getAttribute<ISerializeAttribute>();
    if (serialize)
      serialize->writeToFile(properties[i], object, file);
  }
}

bool Serializer::writeToFile(const capputils::reflection::ReflectableClass& object, const std::string& filename) {
  std::ofstream file(filename.c_str());
  if(!file)
    return false;
    
  writeToFile(object, file);
  file.close();

  return file.good();
}

bool Serializer::readFromFile(capputils::reflection::ReflectableClass& object, std::istream& file) {
  std::vector<capputils::reflection::IClassProperty*>& properties = object.getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    capputils::attributes::ISerializeAttribute* serialize = properties[i]->getAttribute<ISerializeAttribute>();
    if (serialize)
      serialize->readFromFile(properties[i], object, file);
  }
  return true;
}

bool Serializer::readFromFile(capputils::reflection::ReflectableClass& object, const std::string& filename) {
  std::ifstream file(filename.c_str());
  if (!file)
    return false;
    
  readFromFile(object, file);
  file.close();

  return file.good();
}

}
