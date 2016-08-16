/*
 * AbstractEnumerator.cpp
 *
 *  Created on: Mar 8, 2011
 *      Author: tombr
 */

#include <capputils/AbstractEnumerator.h>

#include <stdexcept>

namespace capputils {

AbstractEnumerator::AbstractEnumerator() {
}

AbstractEnumerator::~AbstractEnumerator() {
}

void AbstractEnumerator::toStream(std::ostream& stream) const {
  stream << value;
}

void AbstractEnumerator::fromStream(std::istream& stream) {
  std::vector<std::string>& values = getValues();
  stream >> value;
  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i] == value)
      return;
  }
  throw std::runtime_error("invalid enumerator value");
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
