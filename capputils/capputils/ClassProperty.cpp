/*
 * ClassProperty.cpp
 *
 *  Created on: Mar 8, 2011
 *      Author: tombr
 */

#include "ClassProperty.h"

#include "ReflectableClass.h"

namespace capputils {

namespace reflection {

template<>
std::string convertFromString(const std::string& value) {
  return std::string(value);
}

}

}
