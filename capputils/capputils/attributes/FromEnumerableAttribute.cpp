/*
 * FromEnumerableAttribute.cpp
 *
 *  Created on: May 31, 2011
 *      Author: tombr
 */

#include <capputils/attributes/FromEnumerableAttribute.h>

namespace capputils {

namespace attributes {

FromEnumerableAttribute::FromEnumerableAttribute(int enumerablePropertyId)
 : enumerablePropertyId(enumerablePropertyId)
{
}

FromEnumerableAttribute::~FromEnumerableAttribute() {
}

int FromEnumerableAttribute::getEnumerablePropertyId() const {
  return enumerablePropertyId;
}

AttributeWrapper* FromEnumerable(int enumerablePropertyId) {
  return new AttributeWrapper(new FromEnumerableAttribute(enumerablePropertyId));
}

}

}
