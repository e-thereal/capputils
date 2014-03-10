/*
 * GreaterThanAttribute.h
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_GEATERTHANATTRIBUTE_H_
#define CAPPUTILS_GEATERTHANATTRIBUTE_H_

#include <capputils/attributes/IAssertionAttribute.h>
#include <iostream>
#include <sstream>

#include <capputils/reflection/ClassProperty.h>
#include <capputils/exceptions/AssertionException.h>

namespace capputils {

namespace attributes {

template<class T>
class GreaterThanAttribute: public virtual IAssertionAttribute {
private:
  T value;
  std::string defaultMessage;
  std::string message;

public:
  GreaterThanAttribute(T value, const std::string& defaultMessage = "")
   : value(value), defaultMessage(defaultMessage)
  { }
  virtual ~GreaterThanAttribute() { }

  virtual bool valid(const reflection::IClassProperty& property,
        const reflection::ReflectableClass& object)
  {
    if (!dynamic_cast<const reflection::ClassProperty<T>* >(&property))
      throw capputils::exceptions::AssertionException("Cannot cast property to value type.");

    T propertyValue = dynamic_cast<const reflection::ClassProperty<T>* >(&property)->getValue(object);
    if (propertyValue <= value) {
      if (defaultMessage.size()) {
        message = defaultMessage;
      } else {
        std::stringstream s;
        s << property.getName() << " must be greater than '" << value << "'!";
        message = s.str();
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
AttributeWrapper* GreaterThan(T value, const std::string& defaultMessage = "") {
  return new AttributeWrapper(new GreaterThanAttribute<T>(value, defaultMessage));
}

}

}

#endif /* CAPPUTILS_GEATERTHANATTRIBUTE_H_ */
