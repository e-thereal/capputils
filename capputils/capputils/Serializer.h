#ifndef _CAPPUTILS_SERIALIZER_H_
#define _CAPPUTILS_SERIALIZER_H_

#include "ReflectableClass.h"
#include "ClassProperty.h"

namespace capputils {

class Serializer {
public:
  static void writeToFile(const capputils::reflection::ReflectableClass& object, FILE* file);
  static bool writeToFile(const capputils::reflection::ReflectableClass& object, const std::string& filename);
  static bool readFromFile(capputils::reflection::ReflectableClass& object, FILE* file);
  static bool readFromFile(capputils::reflection::ReflectableClass& object, const std::string& filename);
};

}

#endif /* _CAPPUTILS_SERIALIZER_H_ */