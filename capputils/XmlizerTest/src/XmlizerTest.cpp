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
#include <capputils/GenerateBashCompletion.h>

#include <fstream>
#include <sstream>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/range.hpp>

#include <boost/regex.hpp>
#include <boost/timer.hpp>
#include <memory>

#include "Car.h"
#include "Student.h"

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

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

namespace bio = boost::iostreams;

enum Days {Monday, Tuesday};
CapputilsEnumerator(Tage, Montag, Dienstag);

#define TRACE std::cout << __LINE__ << std::endl;

class LeakTest {
private:
  static std::vector<int> ints;

public:
  void method() {
    static bool initialized = false;
    if (!initialized) {
      ints.push_back(3);
    }
    initialized = true;
  }
};

std::vector<int> LeakTest::ints;

int main(int argc, char** argv) {
  std::cout << capputils::reflection::Converter<unsigned short>::fromString("3+5") << std::endl;

  return 0;

  Days day = Monday;
  Tage tag = Tage::Montag; // tag = 0; tag = "Montag";

  Car car;
  car.Changed.connect(changeHandler);

  std::cout << car << std::endl;


  ReflectableClassFactory& factory = ReflectableClassFactory::getInstance();
  vector<string>& classNames = factory.getClassNames();

  LeakTest test;
  test.method();

  for (unsigned i = 0; i < classNames.size(); ++i) {
    cout << classNames[i] << endl;
    ReflectableClassFactory& factory = ReflectableClassFactory::getInstance();
    ReflectableClass* object = factory.newInstance(classNames[i]);
//    vector<IClassProperty*>& properties = object->getProperties();
//    for (unsigned j = 0; j < properties.size(); ++j)
//      cout << "  - " << properties[j]->getType().name() << " " << properties[j]->getName() << endl;
    delete object;
    cout << endl;
  }
//
//  //Xmlizer::FromXml(car, "car2.xml");
//  ArgumentsParser::Parse(car, argc, argv);
//  if (car.getHelp() || !Verifier::Valid(car)) {
//    ArgumentsParser::PrintUsage(string("\nUsage: ") + argv[0] + " [switches], where switches are:", car);
//    return 0;
//  }
//  boost::shared_ptr<vector<boost::shared_ptr<Person> > > owners = car.getOwners();
//  owners->push_back(boost::shared_ptr<Person>(new Person()));
//  //owners->push_back(boost::shared_ptr<Person>(new Student()));
//
//  cout << "Xmlizer Test" << endl;
//  cout << "Doors: " << car.getProperty("DoorCount") << endl;
//  cout << "High Speed: " << car.getHighSpeed() << endl;
//  cout << "Model Name: " << car.getModelName() << endl;
////	cout << "Engine Type: " << car.getEngine() << endl;
//  //Xmlizer::ToXml("car.xml", car);
//
//  if (car.getGenerateBashCompletion().size()) {
//    GenerateBashCompletion::Generate("XmlizerTest", car, car.getGenerateBashCompletion());
//  }

  std::cout << "Good bye." << std::endl;

  return 0;
}
