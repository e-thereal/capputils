/*
 * FilenameAttribute.cpp
 *
 *  Created on: Feb 11, 2011
 *      Author: tombr
 */

#include "FilenameAttribute.h"

#include <cstdio>

#include "ReflectableClass.h"
#include "ClassProperty.h"

using namespace std;

namespace capputils {

using namespace reflection;

namespace attributes {

FilenameAttribute::FilenameAttribute(const string& pattern, bool multipleSelection)
 : pattern(pattern), multipleSelection(multipleSelection)
{
}

FilenameAttribute::~FilenameAttribute() {
}

bool FilenameAttribute::valid(const IClassProperty& property,
    const ReflectableClass& object)
{
  return true;
}

string FilenameAttribute::getLastMessage() const {
  return lastError;
}

bool FilenameAttribute::getMultipleSelection() const {
  return multipleSelection;
}

const std::string& FilenameAttribute::getPattern() const {
  return pattern;
}

AttributeWrapper* Filename(const string& pattern, bool multipleSelection) {
  return new AttributeWrapper(new FilenameAttribute(pattern, multipleSelection));
}

}

}
