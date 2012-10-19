/*
 * NotEmpty.h
 *
 *  Created on: Jul 26, 2012
 *      Author: tombr
 */

#ifndef CAPPUTILSL_ATTRIBUTES_NOTEMPTYATTRIBUTE_H_
#define CAPPUTILSL_ATTRIBUTES_NOTEMPTYATTRIBUTE_H_

#include "IAssertionAttribute.h"

namespace capputils {

namespace attributes {

template<class T>
class empty_trait {
  typedef T value_t;
public:
  static bool is_empty(const value_t& value) {
    return value.size() == 0;
  }
};

template<class T>
class empty_trait<T*> {
  typedef T* value_t;
public:
  static bool is_empty(const value_t& value) {
    return value && value->size() == 0;
  }
};

template<class T>
class empty_trait<boost::shared_ptr<T> > {
  typedef boost::shared_ptr<T> value_t;
public:
  static bool is_empty(const value_t& value) {
    return value && value->size() == 0;
  }
};

class INotEmptyAttribute : public virtual IAssertionAttribute {
protected:
  std::string message, lastMessage;

public:
  INotEmptyAttribute(const std::string& message = "") : message(message) { }

  virtual std::string getLastMessage() const {
    return lastMessage;
  }
};

template<class T>
class NotEmptyAttribute : public INotEmptyAttribute {
  typedef T value_t;
public:
  NotEmptyAttribute(const std::string& message = "") : INotEmptyAttribute(message) { }

  virtual bool valid(const reflection::IClassProperty& property,
      const reflection::ReflectableClass& object)
  {
    const reflection::ClassProperty<value_t>* prop =
        dynamic_cast<const reflection::ClassProperty<value_t>* >(&property);
    if (prop) {
      if (empty_trait<T>::is_empty(prop->getValue(object))) {
        if (message.size()) {
          lastMessage = message;
        } else {
          lastMessage = "Property '" + property.getName() + "' must not be empty.";
        }
        return false;
      }
    }
    return true;
  }
};

template<class T>
AttributeWrapper* NotEmpty(const std::string& message = "") {
  return new AttributeWrapper(new NotEmptyAttribute<T>(message));
}

}

}

#endif /* CAPPUTILSL_ATTRIBUTES_NOTEMPTYATTRIBUTE_H_ */
