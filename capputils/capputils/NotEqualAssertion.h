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

#include "ClassProperty.h"

namespace capputils {

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

  virtual bool valid(const reflection::IClassProperty& property,
        const reflection::ReflectableClass& object)
  {
    // TODO: check if dynamic_cast fails
    T propertyValue = dynamic_cast<const reflection::ClassProperty<T>* >(&property)->getValue(object);
    if (propertyValue == value) {
      if (defaultMessage.size()) {
        message = defaultMessage;
      } else {
        message = property.getName() +
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
AttributeWrapper* NotEqual(T value, const std::string& defaultMessage = "") {
  return new AttributeWrapper(new NotEqualAttribute<T>(value, defaultMessage));
}

}

}

#endif /* NOTEQUALASSERTION_H_ */
