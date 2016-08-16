/*
 * DescriptionAttribute.cpp
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#include <capputils/attributes/DescriptionAttribute.h>

using namespace std;

namespace capputils {

namespace attributes {

DescriptionAttribute::DescriptionAttribute(const string& description) : description(description) { }

DescriptionAttribute::~DescriptionAttribute() {
}

const string& DescriptionAttribute::getDescription() const {
  return description;
}

AttributeWrapper* Description(const std::string& description) {
  return new AttributeWrapper(new DescriptionAttribute(description));
}

}

}
