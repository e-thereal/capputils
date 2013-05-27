#pragma once

#ifndef _CAPPUTILS_IENUMERABLEATTRIBUTE_H_
#define _CAPPUTILS_IENUMERABLEATTRIBUTE_H_

#include "IAttribute.h"
#include "IClassProperty.h"
#include "IPropertyIterator.h"

#include <string>

namespace capputils {

namespace reflection {

class ReflectableClass;

}

namespace attributes {

class IEnumerableAttribute: public virtual IAttribute {
public:
  virtual boost::shared_ptr<reflection::IPropertyIterator> getPropertyIterator(const reflection::ReflectableClass& object, const reflection::IClassProperty* property) = 0;

  // Deletes the old collection instance (if not null) and creates a new collection instance
  virtual void renewCollection(reflection::ReflectableClass& object, const reflection::IClassProperty* property) = 0;
};

}

}


#endif
