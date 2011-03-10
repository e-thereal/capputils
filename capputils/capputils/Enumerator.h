/*
 * Enumerator.h
 *
 *  Created on: Mar 8, 2011
 *      Author: tombr
 */

#ifndef ENUMERATOR_H_
#define ENUMERATOR_H_

#include "ReflectableClass.h"

#include <vector>
#include <string>

namespace capputils {

namespace reflection {

class Enumerator: public ReflectableClass {

InitAbstractReflectableClass(Enumerator)

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

}

#endif /* ENUMERATOR_H_ */
