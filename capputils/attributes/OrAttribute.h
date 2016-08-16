/*
 * OrAttribute.h
 *
 *  Created on: Mar 6, 2014
 *      Author: tombr
 */

#ifndef CAPPUTILS_ORATTRIBUTE_H_
#define CAPPUTILS_ORATTRIBUTE_H_

#include <capputils/attributes/IAssertionAttribute.h>

namespace capputils {

namespace attributes {

class OrAttribute : public virtual IAssertionAttribute {

private:
  IAttribute *first, *second;
  std::string message, lastMessage;

public:
  OrAttribute(AttributeWrapper* first, AttributeWrapper* second, const std::string& message = "");

  virtual bool valid(const reflection::IClassProperty& property,
        const reflection::ReflectableClass& object);

  virtual std::string getLastMessage() const;
};

AttributeWrapper* Or(AttributeWrapper* first, AttributeWrapper* second, const std::string& message = "");

} /* namespace attributes */

} /* namespace capputils */

#endif /* ORATTRIBUTE_H_ */
