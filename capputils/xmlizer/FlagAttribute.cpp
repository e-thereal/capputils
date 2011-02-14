/*
 * Flag.cpp
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#include "FlagAttribute.h"

namespace xmlizer {

namespace attributes {

FlagAttribute::FlagAttribute() {
}

FlagAttribute::~FlagAttribute() {
}

reflection::AttributeWrapper Flag() {
  return reflection::AttributeWrapper(new FlagAttribute());
}

}

}
