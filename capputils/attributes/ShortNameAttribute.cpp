/*
 * ShortNameAttribute.cpp
 *
 *  Created on: May 6, 2011
 *      Author: tombr
 */

#include <capputils/attributes/ShortNameAttribute.h>

using namespace std;

namespace capputils {

namespace attributes {

ShortNameAttribute::ShortNameAttribute(const string& name) : name(name) {
}

ShortNameAttribute::~ShortNameAttribute() {
}

const std::string& ShortNameAttribute::getName() const {
  return name;
}

AttributeWrapper* ShortName(const string& name) {
  return new AttributeWrapper(new ShortNameAttribute(name));
}

}

}
