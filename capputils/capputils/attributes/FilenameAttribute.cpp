/*
 * FilenameAttribute.cpp
 *
 *  Created on: Feb 11, 2011
 *      Author: tombr
 */

#include <capputils/attributes/FilenameAttribute.h>

#include <cstdio>

#include <capputils/reflection/ReflectableClass.h>
#include <capputils/reflection/ClassProperty.h>

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

const std::string FilenameAttribute::getMatchPattern() const {
  string::size_type first = pattern.find_first_of('(');
  if (first != string::npos) {
    string::size_type second = pattern.find_first_of(')', first + 1);
    if (second != string::npos) {
      return pattern.substr(first + 1, second - first - 1);
    } else {
      return pattern;
    }
  } else {
    return pattern;
  }
}

void FilenameAttribute::setPattern(const std::string& pattern) {
  this->pattern = pattern;
}

AttributeWrapper* Filename(const string& pattern, bool multipleSelection) {
  return new AttributeWrapper(new FilenameAttribute(pattern, multipleSelection));
}

}

}
