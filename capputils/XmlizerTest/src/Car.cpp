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

using namespace capputils::attributes;
using namespace capputils::reflection;

BeginPropertyDefinitions(Car)

  DefineProperty(DoorCount, Description("Number of doors (default = 3)"))
  DefineProperty(HighSpeed)
  DefineProperty(ModelName, NotEqual<std::string>("Audi"))
  DefineProperty(Help, Flag(), Description("Show options"))

EndPropertyDefinitions

Car::Car() : _DoorCount(3), _HighSpeed(100), _ModelName("BMW"), _Help(0) {
}

Car::~Car() {
}
