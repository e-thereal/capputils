/*
 * Flag.cpp
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#include "FlagAttribute.h"

namespace capputils {

namespace attributes {

FlagAttribute::FlagAttribute() {
}

FlagAttribute::~FlagAttribute() {
}

AttributeWrapper* Flag() {
  return new AttributeWrapper(new FlagAttribute());
}

}

}
