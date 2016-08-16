#pragma once

#ifndef CAPPUTILS_DEPRECATEDATTRIBUTE_H_
#define CAPPUTILS_DEPRECATEDATTRIBUTE_H_

#include <capputils/attributes/IXmlableAttribute.h>

namespace capputils {

namespace attributes {

class DeprecatedAttribute : public virtual IXmlableAttribute {
private:
  std::string message;

public:
  DeprecatedAttribute(const std::string& message);

  const std::string& getMessage() const;

  virtual void addToReflectableClassNode(TiXmlElement& node,
      const reflection::ReflectableClass& object) const;
};

AttributeWrapper* Deprecated(const std::string& message);

}

}

#endif /* CAPPUTILS_ATTRIBUTES_DEPRECATEDATTRIBUTE_H_ */
