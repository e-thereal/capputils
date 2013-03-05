#pragma once

#ifndef _CAPPUTILS_IPROPERTYITERATOR_H_
#define _CAPPUTILS_IPROPERTYITERATOR_H_

#include "IClassProperty.h"

namespace capputils {

namespace reflection {

class IPropertyIterator : public virtual IClassProperty {
public:
  virtual void reset() = 0;
  virtual bool eof(const ReflectableClass& object) const = 0;
  virtual void next() = 0;
  virtual void prev() = 0;
  virtual void clear(ReflectableClass& object) = 0;
};

}

}

#endif
