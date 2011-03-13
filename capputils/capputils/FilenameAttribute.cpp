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

FilenameAttribute::FilenameAttribute() {
}

FilenameAttribute::~FilenameAttribute() {
}

bool FilenameAttribute::valid(const IClassProperty& property,
    const ReflectableClass& object)
{
  return true;
}

const string& FilenameAttribute::getLastMessage() const {
  return lastError;
}

AttributeWrapper* Filename() {
  return new AttributeWrapper(new FilenameAttribute());
}

}

}
