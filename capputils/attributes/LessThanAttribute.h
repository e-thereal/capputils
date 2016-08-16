/*
 * LessThanAttribute.h
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_LESSTHANATTRIBUTE_H_
#define CAPPUTILS_LESSTHANATTRIBUTE_H_

#include <capputils/attributes/IAssertionAttribute.h>
#include <iostream>
#include <sstream>

#include <capputils/reflection/ClassProperty.h>
#include <capputils/exceptions/AssertionException.h>

namespace capputils {

namespace attributes {

template<class T>
class LessThanAttribute: public virtual IAssertionAttribute {
private:
  T value;
  std::string defaultMessage;
  std::string message;

public:
  LessThanAttribute(T value, const std::string& defaultMessage = "")
   : value(value), defaultMessage(defaultMessage)
  { }
  virtual ~LessThanAttribute() { }

  virtual bool valid(const reflection::IClassProperty& property,
        const reflection::ReflectableClass& object)
  {
    if (!dynamic_cast<const reflection::ClassProperty<T>* >(&property))
      throw capputils::exceptions::AssertionException("Cannot cast property to value type.");

    T propertyValue = dynamic_cast<const reflection::ClassProperty<T>* >(&property)->getValue(object);
    if (propertyValue >= value) {
      if (defaultMessage.size()) {
        message = defaultMessage;
      } else {
        std::stringstream s;
        s << property.getName() << " must be less than '" << value << "'!";
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
AttributeWrapper* LessThan(T value, const std::string& defaultMessage = "") {
  return new AttributeWrapper(new LessThanAttribute<T>(value, defaultMessage));
}

}

}

#endif /* CAPPUTILS_LESSTHANATTRIBUTE_H_ */
