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
  Property(Weight, double)
  Property(Data, boost::shared_ptr<std::vector<int> >)

};

BeginPropertyDefinitions(SerializeTest)

  DefineProperty(Width, Serialize<TYPE_OF(Width)>())
  DefineProperty(Height, Serialize<TYPE_OF(Height)>())
  DefineProperty(Weight, Serialize<TYPE_OF(Weight)>())
  DefineProperty(Data, Serialize<TYPE_OF(Data)>())

EndPropertyDefinitions

struct EmptyBase {
public:
  virtual ~EmptyBase() { }
};

struct TestStruct : public EmptyBase {
  int a;
  int b;
};

void saveMe(std::ostream& out) {
  out << "I need way way way more data to put into the stream or what?";
}

namespace bio = boost::iostreams;

enum Days {Monday, Tuesday};
ReflectableEnum(Tage, Montag, Dienstag);
DefineEnum(Tage);

int main(int argc, char** argv) {
  Days day = Monday;
  Tage tag = Tage::Montag; // tag = 0; tag = "Montag";

  /*{
    bio::filtering_ostream out;
    out.push(boost::iostreams::gzip_compressor());
    out.push(bio::file_descriptor_sink("test.gz"));
    saveMe(out);
  }

  boost::iostreams::filtering_istream in;
  in.push(boost::iostreams::gzip_decompressor());
  in.push(bio::file_descriptor_source("test.gz"));

  string testString;
  while (!in.eof()) {
    in >> testString;
    cout << testString << endl;
  }*/

  //std::tr1::cmatch res;
  //std::string str = "<h2>Egg prices</h2>";
  //std::regex rx("<h(.)>([^<]+)");
  //if (std::regex_search(str.c_str(), res, rx))
  //  std::cout << res[1] << ". " << res[2] << "\n";

  boost::timer timer;
  std::cout << "Max elapsed: " << timer.elapsed_max() << "s." << std::endl;
  return 0;

  boost::cmatch res;
  std::string str = "segmentation/O145101X_0063200.tif";
  boost::regex rx("(.*)/(.*)01X_.*");
  if (boost::regex_search(str.c_str(), res, rx))
    std::cout << res.format("results1/${2}02.tif") << std::endl;
  else
    std::cout << "Not found!" << std::endl;

  return 0;

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
  test.setWeight(72.6);
  
  boost::shared_ptr<std::vector<int> > data(new std::vector<int>());
  data->push_back(1);
  data->push_back(2);
  data->push_back(3);
  test.setData(data);
  
  Serializer::writeToFile(test, "test.bin");
  
  Serializer::readFromFile(test2, "test.bin");
  cout << test.getWeight() << " == " << test2.getWeight() << endl;
  test2.getData()->push_back(4);
  Serializer::writeToFile(test2, "test2.bin");

  TestStruct st;
  cout << (char*)&st.a - (char*)&st << endl;
  cout << (char*)&st.b - (char*)&st << endl;

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
