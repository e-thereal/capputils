/*
 * OperandAttribute.h
 *
 *  Created on: Feb 5, 2014
 *      Author: tombr
 */

#ifndef CAPPUTILS_OPERANDATTRIBUTE_H_
#define CAPPUTILS_OPERANDATTRIBUTE_H_

#include <capputils/ParameterAttribute.h>

namespace capputils {

namespace attributes {

class OperandAttribute : public ParameterAttribute {
public:
  OperandAttribute(const std::string& name);
};

AttributeWrapper* Operand(const std::string& name);

} /* namespace attributes */

} /* namespace capputils */

#endif /* CAPPUTILS_OPERANDATTRIBUTE_H_ */
