/*
 * AbstractEnumerator.cpp
 *
 *  Created on: Mar 8, 2011
 *      Author: tombr
 */

#include "AbstractEnumerator.h"

namespace capputils {

AbstractEnumerator::AbstractEnumerator() {
}

AbstractEnumerator::~AbstractEnumerator() {
}

void AbstractEnumerator::toStream(std::ostream& stream) const {
  stream << value;
}
void AbstractEnumerator::fromStream(std::istream& stream) {
  // TODO: check if values contains value
  stream >> value;
}

}

std::ostream& operator<< (std::ostream& stream, const capputils::AbstractEnumerator& enumerator) {
  enumerator.toStream(stream);
  return stream;
}

std::istream& operator>> (std::istream& stream, capputils::AbstractEnumerator& enumerator) {
  enumerator.fromStream(stream);
  return stream;
}
