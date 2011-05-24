/*
 * IXmlableAttribute.h
 *
 *  Created on: May 19, 2011
 *      Author: tombr
 */

#ifndef IXMLABLEATTRIBUTE_H_
#define IXMLABLEATTRIBUTE_H_

#include "IAttribute.h"

#include <tinyxml/tinyxml.h>

namespace capputils {

namespace reflection {
  class ReflectableClass;
}

namespace attributes {

class IXmlableAttribute : public virtual IAttribute {
public:
  virtual void addToPropertyNode(TiXmlElement& node,
      const reflection::ReflectableClass& object,
      const reflection::IClassProperty* property) const = 0;
  virtual void getFromPropertyNode(const TiXmlElement& node,
      reflection::ReflectableClass& object,
      reflection::IClassProperty* property) const = 0;
};

}

}

#endif /* IXMLABLEATTRIBUTE_H_ */
