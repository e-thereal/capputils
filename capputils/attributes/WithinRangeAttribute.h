/*
 * WithinRangeAttribute.h
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_WITHINRANGEATTRIBUTE_H_
#define CAPPUTILS_WITHINRANGEATTRIBUTE_H_

#include <capputils/attributes/IAssertionAttribute.h>
#include <iostream>
#include <sstream>

#include <capputils/reflection/ClassProperty.h>
#include <capputils/exceptions/AssertionException.h>

namespace capputils {

namespace attributes {

template<class T>
class WithinRangeAttribute: public virtual IAssertionAttribute {
private:
  T lower, upper;
  std::string defaultMessage;
  std::string message;

public:
  WithinRangeAttribute(T lower, T upper, const std::string& defaultMessage = "")
   : lower(lower), upper(upper), defaultMessage(defaultMessage)
  { }
  virtual ~WithinRangeAttribute() { }

  virtual bool valid(const reflection::IClassProperty& property,
        const reflection::ReflectableClass& object)
  {
    if (!dynamic_cast<const reflection::ClassProperty<T>* >(&property))
      throw capputils::exceptions::AssertionException("Cannot cast property to value type.");
    T propertyValue = dynamic_cast<const reflection::ClassProperty<T>* >(&property)->getValue(object);
    if (propertyValue < lower || propertyValue > upper) {
      if (defaultMessage.size()) {
        message = defaultMessage;
      } else {
        std::stringstream s;
        s << property.getName() << " must be between '" << lower << "' and '" << upper << "'!";
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
AttributeWrapper* WithinRange(T lower, T upper, const std::string& defaultMessage = "") {
  return new AttributeWrapper(new WithinRangeAttribute<T>(lower, upper, defaultMessage));
}

}

}

#endif /* CAPPUTILS_WITHINRANGEATTRIBUTE_H_ */
