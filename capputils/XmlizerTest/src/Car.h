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
#include <boost/shared_ptr.hpp>

#include <capputils/Enumerators.h>

#include "Person.h"

CapputilsEnumerator(EngineType, NoEngine, Diesel, Gas);

class EngineDescription : public capputils::reflection::ReflectableClass {

  InitReflectableClass(EngineDescription)

  Property(CylinderCount, int)
  Property(PS, int)
  Property(Model, EngineType)

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
  Property(LicenceFile, std::string)
  Property(Help, bool)
  Property(Engine, EngineType)
  Property(Owners, boost::shared_ptr<std::vector< boost::shared_ptr<Person> > >)
  Property(SetOnCompilation, int)
  Property(GenerateBashCompletion, std::string)

public:
  Car();
  virtual ~Car();
};

#endif /* CAR_H_ */
