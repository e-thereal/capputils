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

class SerializeTest : public capputils::reflection::ReflectableClass {

  InitReflectableClass(SerializeTest)

  Property(Width, unsigned)
  Property(Height, unsigned)
  Property(Weight, double)
  Property(Data, boost::shared_ptr<std::vector<int> >)
  Property(Data2, boost::weak_ptr<int>)

};

BeginPropertyDefinitions(SerializeTest)

  DefineProperty(Width, Serialize<TYPE_OF(Width)>())
  DefineProperty(Height, Serialize<TYPE_OF(Height)>())
  DefineProperty(Weight, Serialize<TYPE_OF(Weight)>())
  DefineProperty(Data, Serialize<TYPE_OF(Data)>())
  DefineProperty(Data2)

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
CapputilsEnumerator(Tage, Montag, Dienstag);

class IWorkflowElement {
public:
  virtual ~IWorkflowElement() { }

  virtual void startUpdate() const = 0;
  virtual void writeResults() = 0;
};

template<class T>
class WorkflowElement : public virtual IWorkflowElement {

protected:
  mutable T* newState;

public:
  WorkflowElement() : newState(0) { }
  virtual ~WorkflowElement() { if (newState) delete newState; }

  virtual void startUpdate() const {
    if (!newState)
      newState = new T();
  }

  virtual void writeResults() {
    // Go throw the properties and transfer the outputs
  }

protected:
  virtual void update() const = 0;
};

class HelloWorldElement : public WorkflowElement<HelloWorldElement> {
public:
  void sayHello() { }

protected:
  virtual void update() const {
    // do something here

    newState->sayHello();
  }
};

class LogBoard;

class ScopeTest {
private:
  boost::shared_ptr<std::stringstream> s;
  LogBoard* logBoard;

public:
  ScopeTest(LogBoard* logBoard) : s(new std::stringstream()), logBoard(logBoard) {
  }

  virtual ~ScopeTest();

  template<class T>
  std::ostream& operator<<(const T& value) {
    return *s << value;
  }
};

class LogBoard {
private:
  std::string uuid;

public:
  LogBoard(const std::string& uuid) : uuid(uuid) { }

  ScopeTest operator()() {
    return ScopeTest(this);
  }

  void showMessage(const std::string& message) {
    std::cout << uuid << ": " << message << std::endl;
  }
};

ScopeTest::~ScopeTest() {
  logBoard->showMessage(s->str());
}

class A {
public:
  virtual ~A() { }
};

class B : public A { };

int main(int argc, char** argv) {

  std::shared_ptr<A> ptr = std::make_shared<B>();
  std::weak_ptr<A> ptr2 = ptr;
  std::weak_ptr<B> ptr3 = dynamic_pointer_cast<B>(ptr2.lock());

  std::cout << "start scope test." << std::endl;
  LogBoard dlog("ModuleA");
  {
    dlog() << "Hello World " << 2;
    dlog() << "Bla bla bla" << &dlog;
    std::cout << "test." << std::endl;
  }
  std::cout << "end scope test." << std::endl;

  HelloWorldElement element;
  element.startUpdate();
  element.writeResults();

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
