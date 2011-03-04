/*
 * FileExists.cpp
 *
 *  Created on: Feb 11, 2011
 *      Author: tombr
 */

#include "FileExists.h"

#include <cstdio>

#include "ReflectableClass.h"
#include "ClassProperty.h"

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
  const string& filename = dynamic_cast<const ClassProperty<string>*>(&property)->getValue(object);

  FILE* file = fopen(filename.c_str(), "r");
  if (file) {
    fclose(file);
  } else {
    lastError = string("File '") + filename + "' does not exist.";
    return false;
  }

  return true;
}

const string& FileExistsAttribute::getLastMessage() const {
  return lastError;
}

AttributeWrapper* FileExists() {
  return new AttributeWrapper(new FileExistsAttribute());
}

}

}