/*
 * Auto.cpp
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#include "Car.h"

#include <DescriptionAttribute.h>
#include <FlagAttribute.h>
#include <NotEqualAssertion.h>
#include <ObserveAttribute.h>

using namespace capputils::attributes;
using namespace capputils::reflection;

BeginPropertyDefinitions(EngineDescription)
  DefineProperty(CylinderCount)
  DefineProperty(PS)
  DefineProperty(Model)
EndPropertyDefinitions

EngineDescription::EngineDescription() : _CylinderCount(12), _PS(120), _Model(Engine::Gas) { }

EngineDescription::~EngineDescription() { }

BeginPropertyDefinitions(Car)

  DefineProperty(DoorCount, Description("Number of doors (default = 3)"), Observe(PROPERTY_ID))
  DefineProperty(HighSpeed, Observe(PROPERTY_ID))
  DefineProperty(ModelName, NotEqual<std::string>("Audi"), Observe(PROPERTY_ID))
  DefineProperty(Help, Flag(), Description("Show options"), Observe(PROPERTY_ID))
  //DefineProperty(Engine)
  ReflectableProperty(Engine)

EndPropertyDefinitions

Car::Car() : _DoorCount(3), _HighSpeed(100), _ModelName("BMW"), _Help(0)/*, _Engine(Gas)*/ {
}

Car::~Car() {
}
