//============================================================================
// Name        : XmlizerTest.cpp
// Author      : Tom Brosch
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>

#include <Xmlizer.h>
#include <ArgumentsParser.h>
#include <Verifier.h>
#include <ReflectableClassFactory.h>
#include <ClassProperty.h>

#include "Car.h"
#include "Student.h"

using namespace std;
using namespace capputils;
using namespace capputils::reflection;

void changeHandler(ObservableClass* sender, int eventId) {
  ReflectableClass* reflectable = dynamic_cast<ReflectableClass*>(sender);
  if (reflectable) {
    IClassProperty* property = reflectable->getProperties()[eventId];
    cout << reflectable->getClassName() << "::" << property->getName()
         << " changed to " << property->getStringValue(*reflectable) << "." << endl;
  }
}

int main(int argc, char** argv) {
	Car car;
	car.Changed.connect(changeHandler);

  ReflectableClass* object = &car;
  cout << object->getClassName() << endl;

	/*ReflectableClassFactory& factory = ReflectableClassFactory::getInstance();
	vector<string>& classNames = factory.getClassNames();
	for (unsigned i = 0; i < classNames.size(); ++i) {
	  cout << classNames[i] << endl;
	  ReflectableClass* object = ReflectableClassFactory::getInstance().newInstance(classNames[i]);
    vector<IClassProperty*>& properties = object->getProperties();
    for (unsigned j = 0; j < properties.size(); ++j)
      cout << "  - " << properties[j]->getType().name() << " " << properties[j]->getName() << endl;
    delete object;
    cout << endl;
	}*/

	Xmlizer::FromXml(car, "car2.xml");
	ArgumentsParser::Parse(car, argc, argv);

	if (car.getHelp() || !Verifier::Valid(car)) {
	  ArgumentsParser::PrintUsage(string("\nUsage: ") + argv[0] + " [switches], where switches are:", car);
	  return 0;
  }

  //vector<string>* owners = car.getOwners();
  //owners->push_back("Tomble");
  //owners->push_back("Anni");
  //car.setOwners(owners);
  vector<Person*>* owners = car.getOwners();
  owners->push_back(new Person());
  owners->push_back(new Student());
  
	cout << "Xmlizer Test" << endl;
	cout << "Doors: " << car.getProperty("DoorCount") << endl;
	cout << "High Speed: " << car.getHighSpeed() << endl;
	cout << "Model Name: " << car.getModelName() << endl;
//	cout << "Engine Type: " << car.getEngine() << endl;
	Xmlizer::ToXml("car2.xml", car);

	return 0;
}
