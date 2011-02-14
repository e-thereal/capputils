/*
 * DescriptionAttribute.cpp
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#include "DescriptionAttribute.h"

using namespace std;

namespace xmlizer {

namespace attributes {

DescriptionAttribute::DescriptionAttribute(const string& description) : description(description) { }

DescriptionAttribute::~DescriptionAttribute() {
}

const string& DescriptionAttribute::getDescription() const {
  return description;
}

reflection::AttributeWrapper Description(const std::string& description) {
  return reflection::AttributeWrapper(new DescriptionAttribute(description));
}

}

}
