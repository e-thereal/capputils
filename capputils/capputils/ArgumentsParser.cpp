/*
 * ArgumentsParser.cpp
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#include "ArgumentsParser.h"

#include <cstring>
#include <iostream>
#include <cmath>

#include "FlagAttribute.h"
#include "DescriptionAttribute.h"

using namespace std;

namespace capputils {

using namespace reflection;
using namespace attributes;

ArgumentsParser::ArgumentsParser() {
}

ArgumentsParser::~ArgumentsParser() {

}

void ArgumentsParser::Parse(ReflectableClass& object, int argc, char** argv) {
  for (int i = 0; i < argc; ++i) {
    if (!strncmp(argv[i], "--", 2)) {
      IClassProperty* property = object.findProperty(argv[i]+2);
      if (property) {
        if (property->getAttribute<FlagAttribute>())
          property->setStringValue(object, "1");
        else if (i < argc - 1)
          property->setStringValue(object, argv[++i]);
      }
    }
  }
}

void ArgumentsParser::PrintUsage(const string& header, const reflection::ReflectableClass& object) {
  cout << header << endl << endl;
  vector<IClassProperty*>& properties = object.getProperties();

  size_t columnWidth = 0;
  for (unsigned i = 0; i < properties.size(); ++i)
    columnWidth = max(columnWidth, strlen(properties[i]->getName().c_str()));

  for (unsigned i = 0; i < properties.size(); ++i) {
    cout << "  --" << properties[i]->getName();
    for (unsigned j = 0; j < columnWidth - strlen(properties[i]->getName().c_str()); ++j)
      cout << " ";
    DescriptionAttribute* description = properties[i]->getAttribute<DescriptionAttribute>();
    if (description)
      cout << "   " << description->getDescription();
    cout << endl;
  }
  cout << endl;
}

void ArgumentsParser::PrintDefaultUsage(const std::string& programName, const reflection::ReflectableClass& object) {
  ArgumentsParser::PrintUsage(string("\nUsage: ") + programName + " [switches], where switches are:", object);
}

}
