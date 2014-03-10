/*
 * NotAttribute.h
 *
 *  Created on: Mar 6, 2014
 *      Author: tombr
 */

#ifndef CAPPUTILS_NOTATTRIBUTE_H_
#define CAPPUTILS_NOTATTRIBUTE_H_

#include <capputils/attributes/IAssertionAttribute.h>

namespace capputils {

namespace attributes {

class NotAttribute : public virtual IAssertionAttribute {

private:
  IAttribute *first;
  std::string message, lastMessage;

public:
  NotAttribute(AttributeWrapper* first, const std::string& message = "");

  virtual bool valid(const reflection::IClassProperty& property,
        const reflection::ReflectableClass& object);

  virtual std::string getLastMessage() const;
};

AttributeWrapper* Not(AttributeWrapper* first, const std::string& message = "");

} /* namespace attributes */

} /* namespace capputils */

#endif /* CAPPUTILS_NOTATTRIBUTE_H_ */
