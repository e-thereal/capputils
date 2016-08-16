/*
 * EmptyAttribute.h
 *
 *  Created on: Mar 6, 2014
 *      Author: tombr
 */

#ifndef CAPPUTILS_EMPTYATTRIBUTE_H_
#define CAPPUTILS_EMPTYATTRIBUTE_H_

#include <capputils/attributes/IAssertionAttribute.h>

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

class IEmptyAttribute : public virtual IAssertionAttribute {
protected:
  std::string message, lastMessage;

public:
  IEmptyAttribute(const std::string& message = "") : message(message) { }

  virtual std::string getLastMessage() const {
    return lastMessage;
  }
};

template<class T>
class EmptyAttribute : public IEmptyAttribute {
  typedef T value_t;
public:
  EmptyAttribute(const std::string& message = "") : IEmptyAttribute(message) { }

  virtual bool valid(const reflection::IClassProperty& property,
      const reflection::ReflectableClass& object)
  {
    const reflection::ClassProperty<value_t>* prop =
        dynamic_cast<const reflection::ClassProperty<value_t>* >(&property);
    if (prop) {
      if (!empty_trait<T>::is_empty(prop->getValue(object))) {
        if (message.size()) {
          lastMessage = message;
        } else {
          lastMessage = "Property '" + property.getName() + "' must be empty.";
        }
        return false;
      }
    }
    return true;
  }
};

template<class T>
AttributeWrapper* Empty(const std::string& message = "") {
  return new AttributeWrapper(new EmptyAttribute<T>(message));
}

} /* namespace attributes */

} /* namespace capputils */

#endif /* CAPPUTILS_EMPTYATTRIBUTE_H_ */
