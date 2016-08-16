/*
 * IXmlableAttribute.h
 *
 *  Created on: May 19, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_IXMLABLEATTRIBUTE_H_
#define CAPPUTILS_IXMLABLEATTRIBUTE_H_

#include <capputils/attributes/IAttribute.h>

#include <tinyxml/tinyxml.h>

namespace capputils {

namespace reflection {
  class ReflectableClass;
  class IClassProperty;
}

namespace attributes {

class IXmlableAttribute : public virtual IAttribute {
public:
  virtual void addToPropertyNode(TiXmlElement& /*node*/,
      const reflection::ReflectableClass& /*object*/,
      const reflection::IClassProperty* /*property*/) const { }
  virtual void getFromPropertyNode(const TiXmlElement& /*node*/,
      reflection::ReflectableClass& /*object*/,
      reflection::IClassProperty* /*property*/) const { }

  virtual void addToReflectableClassNode(TiXmlElement& /*node*/,
      const reflection::ReflectableClass& /*object*/) const { }
  virtual void getFromReflectableClassNode(const TiXmlElement& /*node*/,
      reflection::ReflectableClass& /*object*/) const { }
};

}

}

#endif /* IXMLABLEATTRIBUTE_H_ */
