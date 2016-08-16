/*
 * Auto.cpp
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#include "Car.h"

#include <capputils/attributes/DescriptionAttribute.h>
#include <capputils/attributes/EnumeratorAttribute.h>
#include <capputils/attributes/FlagAttribute.h>
#include <capputils/attributes/NotEqualAttribute.h>
#include <capputils/attributes/ObserveAttribute.h>
#include <capputils/attributes/EnumerableAttribute.h>
#include <capputils/attributes/ReuseAttribute.h>
#include <capputils/attributes/TimeStampAttribute.h>
#include <capputils/attributes/VolatileAttribute.h>
#include <capputils/attributes/EnumeratorAttribute.h>
#include <capputils/attributes/FilenameAttribute.h>
#include <capputils/attributes/ParameterAttribute.h>
#include <capputils/attributes/OperandAttribute.h>
#include <capputils/attributes/EmptyAttribute.h>
#include <capputils/attributes/OrAttribute.h>
#include <capputils/attributes/FileExistsAttribute.h>
#include <capputils/attributes/AndAttribute.h>
#include <capputils/attributes/LessThanAttribute.h>
#include <capputils/attributes/GreaterThanAttribute.h>
#include <capputils/attributes/WithinRangeAttribute.h>

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

BeginPropertyDefinitions(Car, Description("This class handles stuff related to cars."))

  DefineProperty(DoorCount, Description("Number of doors (default = 3)"), Observe(Id), Parameter("", "d"))
  DefineProperty(HighSpeed, Observe(Id), TimeStamp(Id), Parameter("speed", "s"), Description("High speed."), WithinRange(0.0, 250.0))
  DefineProperty(ModelName, NotEqual<Type>("Audi"), Observe(Id), Parameter("model"))
  DefineProperty(Nicknames, Observe(Id), Parameter("names", "n"), Enumerable<Type, false>(), Description("List of nicknames."))
//  DefineProperty(Nicknames, Observe(Id), Operand("names"), Enumerable<Type, false>(), Description("List of nicknames."))
  DefineProperty(LicenseFile, Filename("License file (*.lic)"), Observe(Id), Parameter("license"), Or(Empty<Type>(), FileExists(), "License file must exist when given."))
  DefineProperty(Help, Flag(), Description("Show options"), Observe(Id), Parameter("help", "h"))
  DefineProperty(Engine, Enumerator<Type>(), Observe(Id), Operand("engine"), Description("The type of the engine."))
  DefineProperty(Owners, Enumerable<Type, true>())
  DefineProperty(SetOnCompilation, TimeStamp(Id), Volatile())
  DefineProperty(GenerateBashCompletion, Filename(), Observe(Id), Parameter("bash"))

EndPropertyDefinitions

Car::Car() : _DoorCount(3), _HighSpeed(100), _ModelName("BMW"), _Help(0),/* _Engine(new EngineDescription()),*/
  _Owners(new vector<boost::shared_ptr<Person> >()), _SetOnCompilation(0)
{
  //TimeStampAttribute* timeStamp = findProperty("SetOnCompilation")->getAttribute<TimeStampAttribute>();
  //timeStamp->setTime(*this, __DATE__" "__TIME__);
}

Car::~Car() {
}
