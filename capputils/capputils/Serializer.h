#ifndef _CAPPUTILS_SERIALIZER_H_
#define _CAPPUTILS_SERIALIZER_H_

#include "ReflectableClass.h"
#include "ClassProperty.h"

#include <iostream>

namespace capputils {

class Serializer {
public:
  static void WriteToFile(const capputils::reflection::ReflectableClass& object,
      capputils::reflection::IClassProperty* prop, std::ostream& file);
  static void WriteToFile(const capputils::reflection::ReflectableClass& object, std::ostream& file);
  static bool WriteToFile(const capputils::reflection::ReflectableClass& object,
      capputils::reflection::IClassProperty* prop, const std::string& filename);
  static bool WriteToFile(const capputils::reflection::ReflectableClass& object, const std::string& filename);

  static void ReadFromFile(capputils::reflection::ReflectableClass& object,
      capputils::reflection::IClassProperty* prop, std::istream& file);
  static bool ReadFromFile(capputils::reflection::ReflectableClass& object, std::istream& file);
  static bool ReadFromFile(capputils::reflection::ReflectableClass& object,
      capputils::reflection::IClassProperty* prop, const std::string& filename);
  static bool ReadFromFile(capputils::reflection::ReflectableClass& object, const std::string& filename);

  /*** will be removed in future releases ***/
  static void writeToFile(const capputils::reflection::ReflectableClass& object, std::ostream& file) {
    WriteToFile(object, file);
  }

  static bool writeToFile(const capputils::reflection::ReflectableClass& object, const std::string& filename) {
    return WriteToFile(object, filename);
  }

  static bool readFromFile(capputils::reflection::ReflectableClass& object, std::istream& file) {
    return ReadFromFile(object, file);
  }

  static bool readFromFile(capputils::reflection::ReflectableClass& object, const std::string& filename) {
    return ReadFromFile(object, filename);
  }
};

}

#endif /* _CAPPUTILS_SERIALIZER_H_ */
