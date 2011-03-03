/*
 * IExecutableAttribute.h
 *
 *  Created on: Mar 2, 2011
 *      Author: tombr
 */

#ifndef IEXECUTABLEATTRIBUTE_H_
#define IEXECUTABLEATTRIBUTE_H_

#include "ReflectableClass.h"
#include "IAttribute.h"

namespace capputils {

namespace attributes {

class IExecutableAttribute : public virtual IAttribute {
public:
  virtual void execute(reflection::ReflectableClass& object) const = 0;
};

}

}

#endif /* IEXECUTABLEATTRIBUTE_H_ */
