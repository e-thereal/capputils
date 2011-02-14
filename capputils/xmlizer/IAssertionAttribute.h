/*
 * IAssertionAttribute.h
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#ifndef IASSERTIONATTRIBUTE_H_
#define IASSERTIONATTRIBUTE_H_

#include "IAttribute.h"

#include "ReflectableClass.h"

namespace xmlizer {

namespace attributes {

class IAssertionAttribute: public virtual reflection::IAttribute {
public:
  virtual bool valid(const reflection::ClassProperty& property,
      const reflection::ReflectableClass& object) = 0;

  virtual const std::string& getLastMessage() const = 0;
};

}

}

#endif /* IASSERTIONATTRIBUTE_H_ */
