/*
 * Enumerator.cpp
 *
 *  Created on: Mar 8, 2011
 *      Author: tombr
 */

#include "Enumerator.h"

namespace capputils {

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

std::ostream& operator<< (std::ostream& stream, const capputils::Enumerator& enumerator) {
  enumerator.toStream(stream);
  return stream;
}

std::istream& operator>> (std::istream& stream, capputils::Enumerator& enumerator) {
  enumerator.fromStream(stream);
  return stream;
}
