/*
 * Flag.h
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#ifndef FLAG_H_
#define FLAG_H_

#include "IAttribute.h"

namespace xmlizer {

namespace attributes {

class FlagAttribute: public virtual reflection::IAttribute {
public:
  FlagAttribute();
  virtual ~FlagAttribute();
};

reflection::AttributeWrapper Flag();

}

}

#endif /* FLAG_H_ */
