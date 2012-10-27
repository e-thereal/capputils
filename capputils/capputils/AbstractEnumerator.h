/*
 * AbstractEnumerator.h
 *
 *  Created on: Mar 8, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_ABSTRACTENUMERATOR_H_
#define CAPPUTILS_ABSTRACTENUMERATOR_H_

#include "capputils.h"

#include <vector>
#include <string>
#include <ostream>
#include <istream>

namespace capputils {

class AbstractEnumerator {
protected:
  std::string value;

public:
  AbstractEnumerator();
  virtual ~AbstractEnumerator();

  virtual std::vector<std::string>& getValues() const = 0;

  virtual void toStream(std::ostream& stream) const;
  virtual void fromStream(std::istream& stream);
  virtual int toInt() const = 0;
};

}

std::ostream& operator<< (std::ostream& stream, const capputils::AbstractEnumerator& enumerator);
std::istream& operator>> (std::istream& stream, capputils::AbstractEnumerator& enumerator);

#endif /* CAPPUTILS_ABSTRACTENUMERATOR_H_ */
