/*
 * Flag.h
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#ifndef FLAG_H_
#define FLAG_H_

#include "IAttribute.h"

namespace capputils {

namespace attributes {

class FlagAttribute: public virtual IAttribute {
public:
  FlagAttribute();
  virtual ~FlagAttribute();
};

AttributeWrapper Flag();

}

}

#endif /* FLAG_H_ */
