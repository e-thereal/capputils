/*
 * IsNullAttribute.h
 *
 *  Created on: Oct 31, 2014
 *      Author: tombr
 */

#ifndef CAPPUTILS_ATTRIBUTES_ISNULLATTRIBUTE_H_
#define CAPPUTILS_ATTRIBUTES_ISNULLATTRIBUTE_H_

#include <capputils/attributes/IAssertionAttribute.h>
#include <capputils/reflection/ReflectableClass.h>
#include <capputils/reflection/ClassProperty.h>

namespace capputils {
namespace attributes {

class IIsNullAttribute : public virtual IAssertionAttribute {
protected:
  std::string message, lastMessage;

public:
  IIsNullAttribute(const std::string& message = "") : message(message) { }

  virtual std::string getLastMessage() const {
    return lastMessage;
  }
};

template<class T>
class IsNullAttribute : public IIsNullAttribute { };

template<class T>
class IsNullAttribute<T*> : public IIsNullAttribute {

  typedef T* value_t;

public:
  IsNullAttribute(const std::string& message = "") : IIsNullAttribute(message) { }

  virtual bool valid(const reflection::IClassProperty& property,
      const reflection::ReflectableClass& object)
  {
    const reflection::ClassProperty<value_t>* prop =
        dynamic_cast<const reflection::ClassProperty<value_t>* >(&property);
    if (prop) {
      if (prop->getValue(object)) {
        if (message.size()) {
          lastMessage = message;
        } else {
          lastMessage = "Property '" + property.getName() + "' must be null.";
        }
        return false;
      }
    }
    return true;
  }
};

template<class T>
class IsNullAttribute<boost::shared_ptr<T> > : public IIsNullAttribute {

  typedef boost::shared_ptr<T> value_t;

public:
  IsNullAttribute(const std::string& message = "") : IIsNullAttribute(message) { }

  virtual bool valid(const reflection::IClassProperty& property,
      const reflection::ReflectableClass& object)
  {
    const reflection::ClassProperty<value_t>* prop =
        dynamic_cast<const reflection::ClassProperty<value_t>* >(&property);
    if (prop) {
      if (prop->getValue(object)) {
        if (message.size()) {
          lastMessage = message;
        } else {
          lastMessage = "Property '" + property.getName() + "' must be null.";
        }
        return false;
      }
    }
    return true;
  }
};

template<class T>
AttributeWrapper* IsNull(const std::string& message = "") {
  return new AttributeWrapper(new IsNullAttribute<T>(message));
}

} /* namespace attributes */

} /* namespace capputils */

#endif /* CAPPUTILS_ATTRIBUTES_ISNULLATTRIBUTE_H_ */
