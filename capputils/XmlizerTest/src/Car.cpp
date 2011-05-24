/*
 * Auto.cpp
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#include "Car.h"

#include <capputils/DescriptionAttribute.h>
#include <capputils/FlagAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/EnumerableAttribute.h>
#include <capputils/ReuseAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/VolatileAttribute.h>

#include <iostream>

using namespace std;
using namespace capputils::attributes;
using namespace capputils::reflection;

BeginPropertyDefinitions(EngineDescription)
  DefineProperty(CylinderCount)
  DefineProperty(PS)
  DefineProperty(Model)
EndPropertyDefinitions

EngineDescription::EngineDescription() : _CylinderCount(12), _PS(120), _Model(Engine::Diesel) {
  cout << "Create EngineDescription" << endl;
}

EngineDescription::~EngineDescription() {
  cout << "Delete EngineDescription" << endl;
}

BeginPropertyDefinitions(Car)

  DefineProperty(DoorCount, Description("Number of doors (default = 3)"), Observe(PROPERTY_ID))
  DefineProperty(HighSpeed, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ModelName, NotEqual<std::string>("Audi"), Observe(PROPERTY_ID))
  DefineProperty(Help, Flag(), Description("Show options"), Observe(PROPERTY_ID))
  ReflectableProperty(Engine, Observe(PROPERTY_ID), Reuse())
  DefineProperty(Owners, Enumerable<vector<Person*>*, true>())
  DefineProperty(SetOnCompilation, TimeStamp(PROPERTY_ID), Volatile())

EndPropertyDefinitions

Car::Car() : _DoorCount(3), _HighSpeed(100), _ModelName("BMW"), _Help(0), _SetOnCompilation(0) {
  _Owners = new vector<Person*>();
  _Engine = new EngineDescription();

  TimeStampAttribute* timeStamp = findProperty("SetOnCompilation")->getAttribute<TimeStampAttribute>();
  timeStamp->setTime(*this, __DATE__" "__TIME__);
}

Car::~Car() {
  delete _Owners;
  delete _Engine;
}
