/*
 * IExecutableAttribute.h
 *
 *  Created on: Mar 2, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_IEXECUTABLEATTRIBUTE_H_
#define CAPPUTILS_IEXECUTABLEATTRIBUTE_H_

#include <capputils/attributes/IAttribute.h>

#include <capputils/reflection/IClassProperty.h>
#include <capputils/reflection/ReflectableClass.h>

namespace capputils {

namespace attributes {

class IExecutableAttribute : public virtual IAttribute {
public:
  virtual void executeBefore(reflection::ReflectableClass& object, const reflection::IClassProperty& property) const = 0;
  virtual void executeAfter(reflection::ReflectableClass& object, const reflection::IClassProperty& property) const = 0;


};

}

}

#endif /* IEXECUTABLEATTRIBUTE_H_ */
