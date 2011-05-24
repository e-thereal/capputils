/*
 * Auto.h
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#ifndef CAR_H_
#define CAR_H_

#include <capputils/Enumerators.h>
#include <capputils/ReflectableClass.h>
#include <capputils/ObservableClass.h>
#include <vector>
#include <string>
#include <capputils/TimedClass.h>

#include <capputils/Enumerators.h>

#include "Person.h"

ReflectableEnum(Engine, NoEngine, Diesel, Gas);

class EngineDescription : public capputils::reflection::ReflectableClass {

  InitReflectableClass(EngineDescription)

  Property(CylinderCount, int)
  Property(PS, int)
  Property(Model, Engine)

public:
  EngineDescription();
  virtual ~EngineDescription();
};

class Car : public capputils::reflection::ReflectableClass,
            public capputils::ObservableClass,
            public capputils::TimedClass
{
  InitReflectableClass(Car)

  Property(DoorCount, int)
  Property(HighSpeed, float)
  Property(ModelName, std::string)
  Property(Help, bool)
  Property(Engine, EngineDescription*)
  Property(Owners, std::vector<Person*>*)
  Property(SetOnCompilation, int)

public:
  Car();
  virtual ~Car();
};

#endif /* CAR_H_ */
