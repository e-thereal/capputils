/*
 * Auto.h
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#ifndef CAR_H_
#define CAR_H_

#include <Enumerators.h>
#include <ReflectableClass.h>
#include <ObservableClass.h>
#include <vector>
#include <string>

#include <Enumerators.h>

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
            public capputils::ObservableClass
{
  InitReflectableClass(Car)

  Property(DoorCount, int)
  Property(HighSpeed, float)
  Property(ModelName, std::string)
  Property(Help, bool)
  Property(Engine, EngineDescription)
  Property(Owners, std::vector<std::string>*)

public:
  Car();
  virtual ~Car();
};

#endif /* CAR_H_ */
