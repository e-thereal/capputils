#include <capputils/attributes/DeprecatedAttribute.h>

#include <capputils/reflection/ReflectableClass.h>

namespace capputils {

namespace attributes {

DeprecatedAttribute::DeprecatedAttribute(const std::string& message) : message(message) { }

const std::string& DeprecatedAttribute::getMessage() const {
  return message;
}

void DeprecatedAttribute::addToReflectableClassNode(TiXmlElement& node,
      const reflection::ReflectableClass& object) const
{
  node.SetAttribute("deprecated", message);
}

AttributeWrapper* Deprecated(const std::string& message) {
  return new AttributeWrapper(new DeprecatedAttribute(message));
}

}

}
