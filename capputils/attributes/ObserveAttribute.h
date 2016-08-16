/*
 * ObserveAttribute.h
 *
 *  Created on: Mar 2, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_OBSERVEATTRIBUTE_H_
#define CAPPUTILS_OBSERVEATTRIBUTE_H_

#include <capputils/attributes/IExecutableAttribute.h>

namespace capputils {

namespace attributes {

class ObserveAttribute : public virtual IExecutableAttribute {
private:
  int eventId;

public:
  ObserveAttribute(int eventId);
  virtual ~ObserveAttribute();

  int getEventId() const;

  virtual void executeBefore(reflection::ReflectableClass& object, const reflection::IClassProperty& property) const;
  virtual void executeAfter(reflection::ReflectableClass& object, const reflection::IClassProperty& property) const;
};

AttributeWrapper* Observe(int eventId);

}

}

#endif /* OBSERVEATTRIBUTE_H_ */
