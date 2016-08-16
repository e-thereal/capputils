/*
 * NotAttribute.cpp
 *
 *  Created on: Mar 6, 2014
 *      Author: tombr
 */

#include <capputils/attributes/NotAttribute.h>

#include <capputils/Verifier.h>

namespace capputils {

namespace attributes {

NotAttribute::NotAttribute(AttributeWrapper* first, const std::string& message)
 : first(first->attribute), message(message)
{
  delete first;
}

bool NotAttribute::valid(const reflection::IClassProperty& property, const reflection::ReflectableClass& object) {
  IAssertionAttribute *firstAssert = dynamic_cast<IAssertionAttribute*>(first);

  if (!firstAssert || !firstAssert->valid(property, object)) {
    return true;
  } else {
    if (message.size()) {
      lastMessage = message;
    } else {
      lastMessage = std::string("Not ") + firstAssert->getLastMessage();
    }
    return false;
  }
}

std::string NotAttribute::getLastMessage() const {
  return lastMessage;
}

AttributeWrapper* Not(AttributeWrapper* first, const std::string& message) {
  return new AttributeWrapper(new NotAttribute(first, message));
}

} /* namespace attributes */

} /* namespace capputils */
