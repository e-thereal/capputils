/*
 * ObservableClass.cpp
 *
 *  Created on: Mar 2, 2011
 *      Author: tombr
 */

#include "ObservableClass.h"

namespace capputils {

ObservableClass::ObservableClass() {
}

ObservableClass::~ObservableClass() {
}

void ObservableClass::fireEvent(int eventId) {
  Changed(this, eventId);
}

}
