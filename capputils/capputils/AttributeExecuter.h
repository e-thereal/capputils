/*
 * AttributeExecuter.h
 *
 *  Created on: Mar 2, 2011
 *      Author: tombr
 */

#ifndef ATTRIBUTEEXECUTER_H_
#define ATTRIBUTEEXECUTER_H_

#include "capputils.h"
#include "ReflectableClass.h"

namespace capputils {

namespace attributes {

class CAPPUTILS_API AttributeExecuter {
public:
  AttributeExecuter();
  virtual ~AttributeExecuter();

  static void ExecuteBefore(reflection::ReflectableClass& object,
      const reflection::IClassProperty& property);
  static void ExecuteAfter(reflection::ReflectableClass& object,
        const reflection::IClassProperty& property);
};

}

}

#endif /* ATTRIBUTEEXECUTER_H_ */
