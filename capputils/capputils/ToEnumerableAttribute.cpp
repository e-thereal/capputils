/*
 * ToEnumerableAttribute.cpp
 *
 *  Created on: Jun 1, 2011
 *      Author: tombr
 */

#include "ToEnumerableAttribute.h"

namespace capputils {

namespace attributes {

ToEnumerableAttribute::ToEnumerableAttribute(int enumerablePropertyId)
 : enumerablePropertyId(enumerablePropertyId)
{
}

ToEnumerableAttribute::~ToEnumerableAttribute() {
}

int ToEnumerableAttribute::getEnumerablePropertyId() const {
  return enumerablePropertyId;
}

AttributeWrapper* ToEnumerable(int enumerablePropertyId) {
  return new AttributeWrapper(new ToEnumerableAttribute(enumerablePropertyId));
}

}

}
