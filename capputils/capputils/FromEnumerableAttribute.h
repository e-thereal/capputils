/*
 * FromEnumerableAttribute.h
 *
 *  Created on: May 31, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_FROMENUMERABLEATTRIBUTE_H_
#define CAPPUTILS_FROMENUMERABLEATTRIBUTE_H_

#include "IAttribute.h"

namespace capputils {

namespace attributes {

class FromEnumerableAttribute : public virtual IAttribute {
private:
  int enumerablePropertyId;

public:
  FromEnumerableAttribute(int enumerablePropertyId);
  virtual ~FromEnumerableAttribute();

  int getEnumerablePropertyId() const;
};

AttributeWrapper* FromEnumerable(int enumerablePropertyId);

}

}

#endif /* CAPPUTILS_FROMENUMERABLEATTRIBUTE_H_ */
