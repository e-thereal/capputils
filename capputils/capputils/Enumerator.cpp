/*
 * Enumerator.cpp
 *
 *  Created on: Mar 8, 2011
 *      Author: tombr
 */

#include "Enumerator.h"

namespace capputils {

namespace reflection {

BeginPropertyDefinitions(Enumerator)
EndPropertyDefinitions

Enumerator::Enumerator() {
}

Enumerator::~Enumerator() {
}

void Enumerator::toStream(std::ostream& stream) const {
  stream << value;
}
void Enumerator::fromStream(std::istream& stream) {
  // TODO: check if values contains value
  stream >> value;
}

}

}
