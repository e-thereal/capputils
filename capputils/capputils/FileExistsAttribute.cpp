/*
 * FileExists.cpp
 *
 *  Created on: Feb 11, 2011
 *      Author: tombr
 */

#include "FileExistsAttribute.h"

#include <cstdio>

#include "ReflectableClass.h"
#include "ClassProperty.h"
#include "EnumerableAttribute.h"

using namespace std;

namespace capputils {

using namespace reflection;

namespace attributes {

FileExistsAttribute::FileExistsAttribute() {
}

FileExistsAttribute::~FileExistsAttribute() {
}

bool FileExistsAttribute::valid(const IClassProperty& property,
    const ReflectableClass& object)
{
  // TODO: handle exception if class property is not of type string

  const ClassProperty<string>* stringProperty = dynamic_cast<const ClassProperty<string>* >(&property);
  if (stringProperty) {
    const string& filename = stringProperty->getValue(object);

    if (!exists(filename)) {
      lastError = string("File '") + filename + "' does not exist.";
      return false;
    }

    return true;
  } else {
    IEnumerableAttribute* enumerable = property.getAttribute<IEnumerableAttribute>();
    if (enumerable) {
      boost::shared_ptr<IPropertyIterator> iter = enumerable->getPropertyIterator(&property);
      for (iter->reset(); !iter->eof(object); iter->next()) {
        const string& filename = iter->getStringValue(object);

        if (!exists(filename)) {
          lastError = string("File '") + filename + "' does not exist.";
          return false;
        }
      }
      return true;
    }
  }
  lastError = string("Property is not of type string and not enumerable!");
  return false;
}

string FileExistsAttribute::getLastMessage() const {
  return lastError;
}

bool FileExistsAttribute::exists(const std::string& filename) {
  FILE* file = fopen(filename.c_str(), "r");
  if (file) {
    fclose(file);
    return true;
  }
  return false;
}

AttributeWrapper* FileExists() {
  return new AttributeWrapper(new FileExistsAttribute());
}

}

}
