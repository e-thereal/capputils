//============================================================================
// Name        : XmlizerTest.cpp
// Author      : Tom Brosch
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>

#include <capputils/Enumerators.h>
#include <capputils/Xmlizer.h>
#include <capputils/ArgumentsParser.h>
#include <capputils/Verifier.h>
#include <capputils/ReflectableClassFactory.h>
#include <capputils/ClassProperty.h>
#include <capputils/IAttribute.h>
#include <capputils/SerializeAttribute.h>
#include <capputils/Serializer.h>

#include "Car.h"
#include "Student.h"

#include <boost/shared_ptr.hpp>

using namespace std;
using namespace capputils;
using namespace capputils::attributes;
using namespace capputils::reflection;

void changeHandler(ObservableClass* sender, int eventId) {
  ReflectableClass* reflectable = dynamic_cast<ReflectableClass*>(sender);
  if (reflectable) {
    IClassProperty* property = reflectable->getProperties()[eventId];
    cout << reflectable->getClassName() << "::" << property->getName()
         << " changed to " << property->getStringValue(*reflectable) << "." << endl;
  }
}

class SerializeTest : public capputils::reflection::ReflectableClass {

  InitReflectableClass(SerializeTest)

  Property(Width, unsigned)
  Property(Height, unsigned)
  Property(Data, boost::shared_ptr<std::vector<int> >)

};

BeginPropertyDefinitions(SerializeTest)

  DefineProperty(Width, Serialize<TYPE_OF(Width)>())
  DefineProperty(Height, Serialize<TYPE_OF(Height)>())
  DefineProperty(Data, Serialize<TYPE_OF(Data)>())

EndPropertyDefinitions

int main(int argc, char** argv) {
  Car car;
  //cout << __DATE__" "__TIME__ << "test" << endl;
  //car.Changed.connect(changeHandler);

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

  SerializeTest test, test2;
  test.setWidth(0xCAFEBABE);
  test.setHeight(0xBADF00D);
  
  boost::shared_ptr<std::vector<int> > data(new std::vector<int>());
  data->push_back(1);
  data->push_back(2);
  data->push_back(3);
  test.setData(data);
  
  Serializer::writeToFile(test, "test.bin");
  
  Serializer::readFromFile(test2, "test.bin");
  test2.getData()->push_back(4);
  Serializer::writeToFile(test2, "test2.bin");

  return 0;

  //Xmlizer::FromXml(car, "car2.xml");
  ArgumentsParser::Parse(car, argc, argv);

  if (car.getHelp() || !Verifier::Valid(car)) {
    ArgumentsParser::PrintUsage(string("\nUsage: ") + argv[0] + " [switches], where switches are:", car);
    return 0;
  }

  boost::shared_ptr<vector<boost::shared_ptr<Person> > > owners = car.getOwners();
  owners->push_back(boost::shared_ptr<Person>(new Person()));
  owners->push_back(boost::shared_ptr<Person>(new Student()));
  
  cout << "Xmlizer Test" << endl;
  cout << "Doors: " << car.getProperty("DoorCount") << endl;
  cout << "High Speed: " << car.getHighSpeed() << endl;
  cout << "Model Name: " << car.getModelName() << endl;
//	cout << "Engine Type: " << car.getEngine() << endl;
  Xmlizer::ToXml("car.xml", car);

  return 0;
}
