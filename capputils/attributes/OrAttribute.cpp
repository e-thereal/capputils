/*
 * OrAttribute.cpp
 *
 *  Created on: Mar 6, 2014
 *      Author: tombr
 */

#include <capputils/attributes/OrAttribute.h>

#include <capputils/Verifier.h>

namespace capputils {

namespace attributes {

OrAttribute::OrAttribute(AttributeWrapper* first, AttributeWrapper* second, const std::string& message)
 : first(first->attribute), second(second->attribute), message(message)
{
  delete first;
  delete second;
}

bool OrAttribute::valid(const reflection::IClassProperty& property, const reflection::ReflectableClass& object) {
  IAssertionAttribute *firstAssert = dynamic_cast<IAssertionAttribute*>(first);
  IAssertionAttribute *secondAssert = dynamic_cast<IAssertionAttribute*>(second);



  if ((!firstAssert || firstAssert->valid(property, object)) || (!secondAssert || secondAssert->valid(property, object))) {
    return true;
  } else {
    if (message.size()) {
      lastMessage = message;
    } else {
      lastMessage = firstAssert->getLastMessage() + " or " + secondAssert->getLastMessage();
    }
    return false;
  }
}

std::string OrAttribute::getLastMessage() const {
  return lastMessage;
}

AttributeWrapper* Or(AttributeWrapper* first, AttributeWrapper* second, const std::string& message) {
  return new AttributeWrapper(new OrAttribute(first, second, message));
}

} /* namespace attributes */

} /* namespace capputils */
