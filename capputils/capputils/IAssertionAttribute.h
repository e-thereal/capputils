/*
 * IAssertionAttribute.h
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_ATTRIBUTES_IASSERTIONATTRIBUTE_H_
#define CAPPUTILS_ATTRIBUTES_IASSERTIONATTRIBUTE_H_

#include "IAttribute.h"
#include <string>

namespace capputils {

namespace reflection {
  class ReflectableClass;
  class IClassProperty;
}

namespace attributes {

class IAssertionAttribute: public virtual IAttribute {
public:
  virtual bool valid(const reflection::IClassProperty& property,
      const reflection::ReflectableClass& object) = 0;

  virtual std::string getLastMessage() const = 0;
};

}

}

#endif /* CAPPUTILS_ATTRIBUTES_IASSERTIONATTRIBUTE_H_ */
