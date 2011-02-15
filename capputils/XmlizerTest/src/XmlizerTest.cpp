//============================================================================
// Name        : XmlizerTest.cpp
// Author      : Tom Brosch
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cstdarg>

#include "Car.h"

#include <Xmlizer.h>
#include <ArgumentsParser.h>
#include <IAssertionAttribute.h>
#include <DescriptionAttribute.h>
#include <Verifier.h>
#include <cstdarg>

#include <boost/signals.hpp>
#include <boost/lambda/lambda.hpp>

using namespace std;
using namespace capputils;
using namespace capputils::reflection;
using namespace capputils::attributes;

void printHello() {
  cout << "Hello" << endl;

  return;
}

int main(int argc, char** argv) {
	Car car;

	//Xmlizer::FromXml(car, "car.xml");
	ArgumentsParser::Parse(car, argc, argv);

	if (car.getHelp() || !Verifier::Valid(car)) {
	  ArgumentsParser::PrintUsage(string("\nUsage: ") + argv[0] + " [switches], where switches are:", car);
	  return 0;
  }

	cout << "Xmlizer Test" << endl;
	cout << "Doors: " << car.getProperty("DoorCount") << endl;
	cout << "High Speed: " << car.getHighSpeed() << endl;
	cout << "Model Name: " << car.getModelName() << endl;
	Xmlizer::ToXml("car2.xml", car);

	return 0;
}
