/*
 * DummyAttribute.h
 *
 *  Created on: Jan 24, 2013
 *      Author: tombr
 */

#ifndef CAPPUTLIS_ATTRIBUTES_DUMMYATTRIBUTE_H_
#define CAPPUTLIS_ATTRIBUTES_DUMMYATTRIBUTE_H_

#include <capputils/attributes/IAttribute.h>

namespace capputils {

namespace attributes {

class DummyAttribute : public virtual IAttribute {
public:
  DummyAttribute(int);
};

AttributeWrapper* Dummy(int value);

} /* namespace attributes */

} /* namespace capputils */

#endif /* CAPPUTLIS_ATTRIBUTES_DUMMYATTRIBUTE_H_ */
