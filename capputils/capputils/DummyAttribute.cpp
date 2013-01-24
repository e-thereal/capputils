/*
 * DummyAttribute.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: tombr
 */

#include "DummyAttribute.h"

namespace capputils {

namespace attributes {

DummyAttribute::DummyAttribute(int) {
}

AttributeWrapper* Dummy(int value) {
  return new AttributeWrapper(new DummyAttribute(value));
}

} /* namespace attributes */

} /* namespace capputils */
