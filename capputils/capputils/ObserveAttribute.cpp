/*
 * ObserveAttribute.cpp
 *
 *  Created on: Mar 2, 2011
 *      Author: tombr
 */

#include "ObserveAttribute.h"
#include "ObservableClass.h"
#include "IReflectableAttribute.h"

namespace capputils {

using namespace reflection;

namespace attributes {

ObserveAttribute::ObserveAttribute(int eventId) : eventId(eventId) {

}

ObserveAttribute::~ObserveAttribute() {
}

void ObserveAttribute::executeBefore(ReflectableClass& object, const IClassProperty& property) const {
  IReflectableAttribute* reflectable = property.getAttribute<IReflectableAttribute>();
  ObservableClass* observable = dynamic_cast<ObservableClass*>(&object);
  if (reflectable && observable) {
    ReflectableClass* child = reflectable->getValuePtr(object, &property);
    ObservableClass* observableChild = dynamic_cast<ObservableClass*>(child);
    if (observableChild)
      observable->removeChild(observableChild);
  }
}

void ObserveAttribute::executeAfter(ReflectableClass& object, const IClassProperty& property) const {
  ObservableClass* observable = dynamic_cast<ObservableClass*>(&object);
  if (observable) {
    IReflectableAttribute* reflectable = property.getAttribute<IReflectableAttribute>();
    if (reflectable) {
      ReflectableClass* child = reflectable->getValuePtr(object, &property);
      ObservableClass* observableChild = dynamic_cast<ObservableClass*>(child);
      if (observableChild)
        observable->addChild(observableChild, eventId);
    }
    observable->fireEvent(eventId);
  }
}

AttributeWrapper* Observe(int eventId) {
  return new AttributeWrapper(new ObserveAttribute(eventId));
}

}

}
