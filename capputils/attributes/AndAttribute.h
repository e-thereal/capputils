/*
 * AndAttribute.h
 *
 *  Created on: Mar 6, 2014
 *      Author: tombr
 */

#ifndef CAPPUTILS_ANDATTRIBUTE_H_
#define CAPPUTILS_ANDATTRIBUTE_H_

#include <capputils/attributes/IAssertionAttribute.h>

namespace capputils {

namespace attributes {

class AndAttribute : public virtual IAssertionAttribute {

private:
  IAttribute *first, *second;
  std::string message, lastMessage;

public:
  AndAttribute(AttributeWrapper* first, AttributeWrapper* second, const std::string& message = "");

  virtual bool valid(const reflection::IClassProperty& property,
        const reflection::ReflectableClass& object);

  virtual std::string getLastMessage() const;
};

AttributeWrapper* And(AttributeWrapper* first, AttributeWrapper* second, const std::string& message = "");

} /* namespace attributes */

} /* namespace capputils */

#endif /* ORATTRIBUTE_H_ */
