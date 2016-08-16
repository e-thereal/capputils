/*
 * NotEqualAttribute.h
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_NOTEQUALATTRIBUTE_H_
#define CAPPUTILS_NOTEQUALATTRIBUTE_H_

#include <capputils/attributes/IAssertionAttribute.h>
#include <iostream>

#include <capputils/reflection/ClassProperty.h>
#include <capputils/exceptions/AssertionException.h>

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
    if (!dynamic_cast<const reflection::ClassProperty<T>* >(&property))
      throw capputils::exceptions::AssertionException("Cannot cast property to value type.");
    T propertyValue = dynamic_cast<const reflection::ClassProperty<T>* >(&property)->getValue(object);
    if (propertyValue == value) {
      if (defaultMessage.size()) {
        message = defaultMessage;
      } else {
        message = property.getName() +
          " must not be equal to '" + property.getStringValue(object) + "'!";
      }
      return false;
    }
    return true;
  }

  virtual std::string getLastMessage() const {
    return message;
  }
};

template<class T>
AttributeWrapper* NotEqual(T value, const std::string& defaultMessage = "") {
  return new AttributeWrapper(new NotEqualAttribute<T>(value, defaultMessage));
}

}

}

#endif /* CAPPUTILS_NOTEQUALATTRIBUTE_H_ */
