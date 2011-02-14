/*
 * FileExists.cpp
 *
 *  Created on: Feb 11, 2011
 *      Author: tombr
 */

#include "FileExists.h"

#include <cstdio>

using namespace reflection;
using namespace std;

namespace xmlizer {

namespace attributes {

FileExistsAttribute::FileExistsAttribute() {
}

FileExistsAttribute::~FileExistsAttribute() {
}

bool FileExistsAttribute::valid(const ClassProperty& property,
    const ReflectableClass& object)
{
  const string& filename = property.getValue<string>(object);

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

reflection::AttributeWrapper FileExists() {
  return reflection::AttributeWrapper(new FileExistsAttribute());
}

}

}
