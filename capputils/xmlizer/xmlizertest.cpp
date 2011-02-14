// cppxmlizer.cpp : Defines the entry point for the console application.
//

#include <cstdlib>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>

#include "ReflectableClass.h"

using namespace std;
using namespace reflection;

/*** Here comes the true story ***/

/*template<>
float convert(const std::string& value) {
  return 1.0;
}*/

class Test : public reflection::ReflectableClass {
  InitReflectableClass(Test)

  Property(A, int)
  Property(Height, float)
};

BeginPropertyDefinitions(Test)

  DefineProperty(A)
  DefineProperty(Height)

EndPropertyDefinitions

int main(int argc, char* argv[])
{
  Test t;
  t.setA(3);
  t.setHeight(1.74);

  t.setProperty("A", "12");
  t.setProperty("Height", "2.10");
  vector<ClassProperty*>& properties = t.getProperties();
  for (unsigned i = 0; i < properties.size(); ++i)
    cout << properties[i]->name << " = " << t.getProperty(properties[i]->name) << endl;
  //t.printProperties();
  cout << t.getA() << endl;
  cout << t.getHeight() << endl;

  system("pause");
	return 0;
}
