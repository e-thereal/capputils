/*
 * NoParameterAttribute.h
 *
 *  Created on: Jan 23, 2012
 *      Author: tombr
 */

#ifndef CAPPUTILS_ATTRIBUTES_NOPARAMETERATTRIBUTE_H_
#define CAPPUTILS_ATTRIBUTES_NOPARAMETERATTRIBUTE_H_

#include "IAttribute.h"

namespace capputils {

namespace attributes {

class NoParameterAttribute : public virtual IAttribute {
public:
  NoParameterAttribute();
  virtual ~NoParameterAttribute();
};

AttributeWrapper* NoParameter();

} /* namespace attributes */

} /* namespace capputils */

#endif /* CAPPUTILS_ATTRIBUTES_NOPARAMETERATTRIBUTE_H_ */
