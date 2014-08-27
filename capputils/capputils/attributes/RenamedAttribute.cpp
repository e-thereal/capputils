/*
 * RenamedAttribute.cpp
 *
 *  Created on: Aug 26, 2014
 *      Author: tombr
 */

#include "RenamedAttribute.h"

namespace capputils {

namespace attributes {

RenamedAttribute::RenamedAttribute(const std::string& newName) : newName(newName) { }

RenamedAttribute::~RenamedAttribute() { }

const std::string& RenamedAttribute::getNewName() const {
  return newName;
}

AttributeWrapper* Renamed(const std::string& newName) {
  return new AttributeWrapper(new RenamedAttribute(newName));
}

} /* namespace attributes */

} /* namespace capputils */
