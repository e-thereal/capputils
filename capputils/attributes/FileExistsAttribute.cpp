/*
 * FileExists.cpp
 *
 *  Created on: Feb 11, 2011
 *      Author: tombr
 */

#include <capputils/attributes/FileExistsAttribute.h>

#include <cstdio>

#include <capputils/reflection/ReflectableClass.h>
#include <capputils/reflection/ClassProperty.h>
#include <capputils/attributes/EnumerableAttribute.h>

#include <capputils/exceptions/AssertionException.h>

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
  // TODO: Instead, iterate through enumeration if EnumerableAttribute is set
  //if (!dynamic_cast<const ClassProperty<string>* >(&property))
    //throw capputils::exceptions::AssertionException("Cannot cast property to string.");

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
      boost::shared_ptr<IPropertyIterator> iter = enumerable->getPropertyIterator(object, &property);
      for (iter->reset(); !iter->eof(); iter->next()) {
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
