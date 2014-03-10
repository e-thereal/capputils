/*
 * AndAttribute.cpp
 *
 *  Created on: Mar 6, 2014
 *      Author: tombr
 */

#include <capputils/attributes/AndAttribute.h>

#include <capputils/Verifier.h>

namespace capputils {

namespace attributes {

AndAttribute::AndAttribute(AttributeWrapper* first, AttributeWrapper* second, const std::string& message)
 : first(first->attribute), second(second->attribute), message(message)
{
  delete first;
  delete second;
}

bool AndAttribute::valid(const reflection::IClassProperty& property, const reflection::ReflectableClass& object) {
  IAssertionAttribute *firstAssert = dynamic_cast<IAssertionAttribute*>(first);
  IAssertionAttribute *secondAssert = dynamic_cast<IAssertionAttribute*>(second);

  if (firstAssert && !firstAssert->valid(property, object)) {
    lastMessage = (message.size() ? message : firstAssert->getLastMessage());
    return false;
  } else if (secondAssert && !secondAssert->valid(property, object)) {
    lastMessage = (message.size() ? message : secondAssert->getLastMessage());
    return false;
  } else {
    return true;
  }
}

std::string AndAttribute::getLastMessage() const {
  return lastMessage;
}

AttributeWrapper* And(AttributeWrapper* first, AttributeWrapper* second, const std::string& message) {
  return new AttributeWrapper(new AndAttribute(first, second, message));
}

} /* namespace attributes */

} /* namespace capputils */
