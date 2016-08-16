/*
 * TimedClass.h
 *
 *  Created on: May 19, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_TIMEDCLASS_H_
#define CAPPUTILS_TIMEDCLASS_H_

#include <map>
#include <ctime>

namespace capputils {

class TimedClass {
private:
  std::map<int, time_t> times;

public:
  TimedClass();
  virtual ~TimedClass();

  void setCurrentTime(int propertyId);
  void setTime(int propertyId, time_t time);
  time_t getTime(int propertyId) const;
};

}

#endif /* CAPPUTILS_TIMEDCLASS_H_ */
