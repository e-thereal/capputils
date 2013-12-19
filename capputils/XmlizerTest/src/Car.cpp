/*
 * Auto.cpp
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#include "Car.h"

#include <capputils/DescriptionAttribute.h>
#include <capputils/EnumeratorAttribute.h>
#include <capputils/FlagAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/EnumerableAttribute.h>
#include <capputils/ReuseAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/EnumeratorAttribute.h>
#include <capputils/FilenameAttribute.h>

#include <iostream>

using namespace std;
using namespace capputils::attributes;
using namespace capputils::reflection;

BeginPropertyDefinitions(EngineDescription)
  DefineProperty(CylinderCount)
  DefineProperty(PS)
  DefineProperty(Model, Enumerator<EngineType>())
EndPropertyDefinitions

EngineDescription::EngineDescription() : _CylinderCount(12), _PS(120), _Model(EngineType::Diesel) {
  cout << "Create EngineDescription" << endl;
}

EngineDescription::~EngineDescription() {
  cout << "Delete EngineDescription" << endl;
}

BeginPropertyDefinitions(Car)

  DefineProperty(DoorCount, Description("Number of doors (default = 3)"), Observe(Id))
  DefineProperty(HighSpeed, Observe(Id), TimeStamp(Id))
  DefineProperty(ModelName, NotEqual<Type>("Audi"), Observe(Id))
  DefineProperty(LicenceFile, Filename(), Observe(Id))
  DefineProperty(Help, Flag(), Description("Show options"), Observe(Id))
  DefineProperty(Engine, Enumerator<Type>(), Observe(Id))
  DefineProperty(Owners, Enumerable<Type, true>())
  DefineProperty(SetOnCompilation, TimeStamp(Id), Volatile())
  DefineProperty(GenerateBashCompletion, Filename(), Observe(Id))

EndPropertyDefinitions

Car::Car() : _DoorCount(3), _HighSpeed(100), _ModelName("BMW"), _Help(0),/* _Engine(new EngineDescription()),*/
  _Owners(new vector<boost::shared_ptr<Person> >()), _SetOnCompilation(0)
{
  //TimeStampAttribute* timeStamp = findProperty("SetOnCompilation")->getAttribute<TimeStampAttribute>();
  //timeStamp->setTime(*this, __DATE__" "__TIME__);
}

Car::~Car() {
}
