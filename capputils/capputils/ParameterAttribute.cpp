/*
 * ParameterAttribute.cpp
 *
 *  Created on: Dec 18, 2013
 *      Author: tombr
 */

#include "ParameterAttribute.h"

namespace capputils {

namespace attributes {

ParameterAttribute::ParameterAttribute(const std::string& longName, const std::string& shortName)
 : longName(longName), shortName(shortName)
{
}

std::string ParameterAttribute::getLongName() const {
  return longName;
}

std::string ParameterAttribute::getShortName() const {
  return shortName;
}

AttributeWrapper* Parameter(const std::string& longName, const std::string& shortName) {
  return new AttributeWrapper(new ParameterAttribute(longName, shortName));
}

} /* namespace attributes */

} /* namespace capputils */
