#include "Serializer.h"

#include "SerializeAttribute.h"
#include <fstream>

using namespace capputils::attributes;

namespace capputils {

void Serializer::WriteToFile(const capputils::reflection::ReflectableClass& object,
      capputils::reflection::IClassProperty* prop, std::ostream& file)
{
  capputils::attributes::ISerializeAttribute* serialize = prop->getAttribute<ISerializeAttribute>();
  if (serialize)
    serialize->writeToFile(prop, object, file);
}

void Serializer::WriteToFile(const capputils::reflection::ReflectableClass& object, std::ostream& file) {
  std::vector<capputils::reflection::IClassProperty*>& properties = object.getProperties();
  for (unsigned i = 0; i < properties.size(); ++i)
    WriteToFile(object, properties[i], file);
}

bool Serializer::WriteToFile(const capputils::reflection::ReflectableClass& object,
    capputils::reflection::IClassProperty* prop, const std::string& filename)
{
  std::ofstream file(filename.c_str());
  if(!file)
    return false;

  WriteToFile(object, prop, file);
  file.close();

  return file.good();
}

bool Serializer::WriteToFile(const capputils::reflection::ReflectableClass& object, const std::string& filename) {
  std::ofstream file(filename.c_str());
  if(!file)
    return false;
    
  WriteToFile(object, file);
  file.close();

  return file.good();
}

void Serializer::ReadFromFile(capputils::reflection::ReflectableClass& object,
    capputils::reflection::IClassProperty* prop, std::istream& file)
{
  capputils::attributes::ISerializeAttribute* serialize = prop->getAttribute<ISerializeAttribute>();
  if (serialize && !file.eof())
    serialize->readFromFile(prop, object, file);
}

bool Serializer::ReadFromFile(capputils::reflection::ReflectableClass& object, std::istream& file) {
  std::vector<capputils::reflection::IClassProperty*>& properties = object.getProperties();
  for (unsigned i = 0; i < properties.size(); ++i)
    ReadFromFile(object, properties[i], file);

  return true;
}

bool Serializer::ReadFromFile(capputils::reflection::ReflectableClass& object,
      capputils::reflection::IClassProperty* prop, const std::string& filename)
{
  std::ifstream file(filename.c_str());
  if (!file)
    return false;

  ReadFromFile(object, prop, file);
  file.close();

  return file.good();
}

bool Serializer::ReadFromFile(capputils::reflection::ReflectableClass& object, const std::string& filename) {
  std::ifstream file(filename.c_str());
  if (!file)
    return false;
    
  ReadFromFile(object, file);
  file.close();

  return file.good();
}

}
