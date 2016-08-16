/*
 * EqualAttribute.h
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_EQUALATTRIBUTE_H_
#define CAPPUTILS_EQUALATTRIBUTE_H_

#include <capputils/attributes/IAssertionAttribute.h>
#include <iostream>

#include <capputils/reflection/ClassProperty.h>

namespace capputils {

namespace attributes {

template<class T>
class EqualAttribute : public virtual IAssertionAttribute {
private:
  T value;
  std::string defaultMessage;
  std::string message;

public:
  EqualAttribute(T value, const std::string& defaultMessage = "")
   : value(value), defaultMessage(defaultMessage)
  { }
  virtual ~EqualAttribute() { }

  virtual bool valid(const reflection::IClassProperty& property,
        const reflection::ReflectableClass& object)
  {

    T propertyValue = dynamic_cast<const reflection::ClassProperty<T>* >(&property)->getValue(object);
    if (propertyValue != value) {
      if (defaultMessage.size()) {
        message = defaultMessage;
      } else {
        message = property.getName() +
          " must be equal to '" + property.getStringValue(object) + "'!";
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
AttributeWrapper* Equal(T value, const std::string& defaultMessage = "") {
  return new AttributeWrapper(new EqualAttribute<T>(value, defaultMessage));
}

}

}

#endif /* CAPPUTILS_NOTEQUALATTRIBUTE_H_ */
