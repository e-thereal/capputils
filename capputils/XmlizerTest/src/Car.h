/*
 * Auto.h
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#ifndef CAR_H_
#define CAR_H_

#include <ReflectableClass.h>

class Car : public capputils::reflection::ReflectableClass {

  InitReflectableClass(Car)

  Property(DoorCount, int)
  Property(HighSpeed, float)
  Property(ModelName, std::string)
  Property(Help, bool)

public:
  Car();
  virtual ~Car();
};

#endif /* CAR_H_ */
