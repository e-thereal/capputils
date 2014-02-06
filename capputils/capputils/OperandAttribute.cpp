/*
 * OperandAttribute.cpp
 *
 *  Created on: Feb 5, 2014
 *      Author: tombr
 */

#include "OperandAttribute.h"

namespace capputils {

namespace attributes {

OperandAttribute::OperandAttribute(const std::string& name) : ParameterAttribute(name) { }

AttributeWrapper* Operand(const std::string& name) {
  return new AttributeWrapper(new OperandAttribute(name));
}

} /* namespace attributes */

} /* namespace capputils */
