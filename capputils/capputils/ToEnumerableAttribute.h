/*
 * ToEnumerableAttribute.h
 *
 *  Created on: Jun 1, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_TOENUMERABLEATTRIBUTE_H_
#define CAPPUTILS_TOENUMERABLEATTRIBUTE_H_

#include "IAttribute.h"

namespace capputils {

namespace attributes {

class ToEnumerableAttribute : public virtual IAttribute {
private:
  int enumerablePropertyId;

public:
  ToEnumerableAttribute(int enumerablePropertyId);
  virtual ~ToEnumerableAttribute();

  int getEnumerablePropertyId() const;
};

AttributeWrapper* ToEnumerable(int enumerablePropertyId);

}

}

#endif /* CAPPUTILS_TOENUMERABLEATTRIBUTE_H_ */
