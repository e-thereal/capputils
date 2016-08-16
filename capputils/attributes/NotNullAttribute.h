/*
 * NotNullAttribute.h
 *
 *  Created on: Jul 25, 2012
 *      Author: tombr
 */

#ifndef CAPPUTILS_ATTRIBUTES_NOTNULLATTRIBUTE_H_
#define CAPPUTILS_ATTRIBUTES_NOTNULLATTRIBUTE_H_

#include <capputils/attributes/IAssertionAttribute.h>
#include <capputils/reflection/ReflectableClass.h>
#include <capputils/reflection/ClassProperty.h>

namespace capputils {
namespace attributes {

class INotNullAttribute : public virtual IAssertionAttribute {
protected:
  std::string message, lastMessage;

public:
  INotNullAttribute(const std::string& message = "") : message(message) { }

  virtual std::string getLastMessage() const {
    return lastMessage;
  }
};

template<class T>
class NotNullAttribute : public INotNullAttribute { };

template<class T>
class NotNullAttribute<T*> : public INotNullAttribute {

  typedef T* value_t;

public:
  NotNullAttribute(const std::string& message = "") : INotNullAttribute(message) { }

  virtual bool valid(const reflection::IClassProperty& property,
      const reflection::ReflectableClass& object)
  {
    const reflection::ClassProperty<value_t>* prop =
        dynamic_cast<const reflection::ClassProperty<value_t>* >(&property);
    if (prop) {
      if (!prop->getValue(object)) {

        if (message.size()) {
          lastMessage = message;
        } else {
          lastMessage = "Property '" + property.getName() + "' must not be null.";
        }
        return false;
      }
    }
    return true;
  }
};

template<class T>
class NotNullAttribute<boost::shared_ptr<T> > : public INotNullAttribute {

  typedef boost::shared_ptr<T> value_t;

public:
  NotNullAttribute(const std::string& message = "") : INotNullAttribute(message) { }

  virtual bool valid(const reflection::IClassProperty& property,
      const reflection::ReflectableClass& object)
  {
    const reflection::ClassProperty<value_t>* prop =
        dynamic_cast<const reflection::ClassProperty<value_t>* >(&property);
    if (prop) {
      if (!prop->getValue(object)) {

        if (message.size()) {
          lastMessage = message;
        } else {
          lastMessage = "Property '" + property.getName() + "' must not be null.";
        }
        return false;
      }
    }
    return true;
  }
};

template<class T>
AttributeWrapper* NotNull(const std::string& message = "") {
  return new AttributeWrapper(new NotNullAttribute<T>(message));
}

} /* namespace attributes */

} /* namespace capputils */

#endif /* CAPPUTILS_ATTRIBUTES_NOTNULLATTRIBUTE_H_ */
