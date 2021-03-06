/*
 * TimedClass.cpp
 *
 *  Created on: May 19, 2011
 *      Author: tombr
 */

#include <capputils/TimedClass.h>

namespace capputils {

TimedClass::TimedClass() {
}

TimedClass::~TimedClass() {
}

void TimedClass::setCurrentTime(int propertyId) {
  setTime(propertyId, std::time(0));
}

void TimedClass::setTime(int propertyId, time_t time) {
  times[propertyId] = time;
}

time_t TimedClass::getTime(int propertyId) const {
  if (times.find(propertyId) != times.end())
    return times.at(propertyId);
  return 0;
}

}
