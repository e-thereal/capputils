/*
 * NotEqualAssertion.h
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#ifndef NOTEQUALASSERTION_H_
#define NOTEQUALASSERTION_H_

#include "IAssertionAttribute.h"
#include <iostream>


namespace xmlizer {

namespace attributes {

template<class T>
class NotEqualAttribute: public virtual IAssertionAttribute {
private:
  T value;
  std::string defaultMessage;
  std::string message;

public:
  NotEqualAttribute(T value, const std::string& defaultMessage = "")
   : value(value), defaultMessage(defaultMessage)
  { }
  virtual ~NotEqualAttribute() { }

  virtual bool valid(const reflection::ClassProperty& property,
        const reflection::ReflectableClass& object)
  {
    const std::string& propertyValue = property.getRawValue(object);
    if (reflection::convertFromString<T>(propertyValue) == value) {
      if (defaultMessage.size()) {
        message = defaultMessage;
      } else {
        message = property.name +
            " must not be " + propertyValue + "!";
      }
      return false;
    }
    return true;
  }

  virtual const std::string& getLastMessage() const {
    return message;
  }
};

template<class T>
reflection::AttributeWrapper NotEqual(T value, const std::string& defaultMessage = "") {
  return reflection::AttributeWrapper(new NotEqualAttribute<T>(value, defaultMessage));
}

}

}

#endif /* NOTEQUALASSERTION_H_ */
