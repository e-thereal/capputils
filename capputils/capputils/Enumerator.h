/*
 * Enumerator.h
 *
 *  Created on: Mar 8, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_ENUMERATOR_H_
#define CAPPUTILS_ENUMERATOR_H_

#include "capputils.h"

#include <vector>
#include <string>
#include <ostream>
#include <istream>

namespace capputils {

class Enumerator {
protected:
  std::string value;

public:
  Enumerator();
  virtual ~Enumerator();

  virtual std::vector<std::string>& getValues() const = 0;

  virtual void toStream(std::ostream& stream) const;
  virtual void fromStream(std::istream& stream);
  virtual int toInt() const = 0;
};

}

std::ostream& operator<< (std::ostream& stream, const capputils::Enumerator& enumerator);
std::istream& operator>> (std::istream& stream, capputils::Enumerator& enumerator);

#endif /* CAPPUTILS_ENUMERATOR_H_ */
