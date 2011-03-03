/*
 * ObserveAttribute.h
 *
 *  Created on: Mar 2, 2011
 *      Author: tombr
 */

#ifndef OBSERVEATTRIBUTE_H_
#define OBSERVEATTRIBUTE_H_

#include "IExecutableAttribute.h"

namespace capputils {

namespace attributes {

class ObserveAttribute : public virtual IExecutableAttribute {
private:
  int eventId;

public:
  ObserveAttribute(int eventId);
  virtual ~ObserveAttribute();

  virtual void execute(reflection::ReflectableClass& object) const;
};

AttributeWrapper* Observe(int eventId);

}

}

#endif /* OBSERVEATTRIBUTE_H_ */
