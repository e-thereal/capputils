/*
 * NoParameterAttribute.cpp
 *
 *  Created on: Jan 23, 2012
 *      Author: tombr
 */

#include "NoParameterAttribute.h"

namespace capputils {

namespace attributes {

NoParameterAttribute::NoParameterAttribute() {

}

NoParameterAttribute::~NoParameterAttribute() {
}

AttributeWrapper* NoParameter() {
  return new AttributeWrapper(new NoParameterAttribute());
}

} /* namespace attributes */

} /* namespace capputils */
