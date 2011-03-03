/*
 * AttributeExecuter.h
 *
 *  Created on: Mar 2, 2011
 *      Author: tombr
 */

#ifndef ATTRIBUTEEXECUTER_H_
#define ATTRIBUTEEXECUTER_H_

#include "ReflectableClass.h"

namespace capputils {

namespace attributes {

class AttributeExecuter {
public:
  AttributeExecuter();
  virtual ~AttributeExecuter();

  static void Execute(reflection::ReflectableClass& object,
      const reflection::IClassProperty& property);
};

}

}

#endif /* ATTRIBUTEEXECUTER_H_ */
