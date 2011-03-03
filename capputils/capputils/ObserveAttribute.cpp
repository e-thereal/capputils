/*
 * ObserveAttribute.cpp
 *
 *  Created on: Mar 2, 2011
 *      Author: tombr
 */

#include "ObserveAttribute.h"
#include "ObservableClass.h"

namespace capputils {

namespace attributes {

ObserveAttribute::ObserveAttribute(int eventId) : eventId(eventId) {

}

ObserveAttribute::~ObserveAttribute() {
}

void ObserveAttribute::execute(reflection::ReflectableClass& object) const {
  ObservableClass* observable = dynamic_cast<ObservableClass*>(&object);
  if (observable) {
    observable->fireEvent(eventId);
  }
}

AttributeWrapper* Observe(int eventId) {
  return new AttributeWrapper(new ObserveAttribute(eventId));
}

}

}
