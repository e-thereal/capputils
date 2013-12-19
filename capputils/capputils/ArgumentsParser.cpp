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
#include "HideAttribute.h"
#include "ParameterAttribute.h"

using namespace std;

namespace capputils {

using namespace reflection;
using namespace attributes;

void ArgumentsParser::Parse(ReflectableClass& object, int argc, char** argv, bool parseOnlyParameter) {
  std::vector<IClassProperty*>& properties = object.getProperties();
  ParameterAttribute* parameter = NULL;

  for (int iArg = 0; iArg < argc; ++iArg) {

    if (!strncmp(argv[iArg], "--", 2)) {

      // long parameter name
      for (size_t iProp = 0; iProp < properties.size(); ++iProp) {
        parameter = properties[iProp]->getAttribute<ParameterAttribute>();

        if (properties[iProp]->getAttribute<HideAttribute>() || (parseOnlyParameter && !parameter))
          continue;

        if ((parameter && parameter->getLongName().compare(argv[iArg] + 2) == 0) ||
            (!parameter && properties[iProp]->getName().compare(argv[iArg] + 2) == 0))
        {
          if (properties[iProp]->getAttribute<FlagAttribute>())
            properties[iProp]->setStringValue(object, "1");
          else if (iArg < argc - 1)
            properties[iProp]->setStringValue(object, argv[++iArg]);
          break;
        }
      }
    } else if (!strncmp(argv[iArg], "-", 1)) {

      // Short parameter name
      for (size_t iProp = 0; iProp < properties.size(); ++iProp) {
        parameter = properties[iProp]->getAttribute<ParameterAttribute>();

        if (properties[iProp]->getAttribute<HideAttribute>() || !parameter)
          continue;

        if (parameter->getShortName().compare(argv[iArg] + 1) == 0) {
          if (properties[iProp]->getAttribute<FlagAttribute>())
            properties[iProp]->setStringValue(object, "1");
          else if (iArg < argc - 1)
            properties[iProp]->setStringValue(object, argv[++iArg]);
          break;
        }
      }
    } else {
      // default parameter
    }
  }
}

void ArgumentsParser::PrintUsage(const string& header, const reflection::ReflectableClass& object, bool showOnlyParameters) {
  cout << header << endl << endl;
  vector<IClassProperty*>& properties = object.getProperties();
  ParameterAttribute* parameter = NULL;

  size_t column1Width = 0;
  for (size_t i = 0; i < properties.size(); ++i) {
    parameter = properties[i]->getAttribute<ParameterAttribute>();
    if (properties[i]->getAttribute<HideAttribute>() || (showOnlyParameters && !parameter))
      continue;
    if (parameter)
      column1Width = max(column1Width, parameter->getShortName().size());
  }

  size_t column2Width = 0;
  for (size_t i = 0; i < properties.size(); ++i) {
    parameter = properties[i]->getAttribute<ParameterAttribute>();
    if (properties[i]->getAttribute<HideAttribute>() || (showOnlyParameters && !parameter))
      continue;
    if (parameter)
      column2Width = max(column2Width, parameter->getLongName().size());
    else
      column2Width = max(column2Width, properties[i]->getName().size());
  }

  for (size_t i = 0; i < properties.size(); ++i) {
    parameter = properties[i]->getAttribute<ParameterAttribute>();
    if (properties[i]->getAttribute<HideAttribute>() || (showOnlyParameters && !parameter))
      continue;
    std::string shortName = (parameter ? parameter->getShortName() : "");
    std::string fullName = (parameter ? parameter->getLongName() : properties[i]->getName());
    if (shortName.size())
      cout << "  -" << shortName;
    else
      cout << "   ";
    for (size_t j = 0; j < column1Width - shortName.size(); ++j)
      cout << " ";
    if (fullName.size())
      cout << " --" << fullName;
    else
      cout << "   ";
    for (size_t j = 0; j < column2Width - fullName.size(); ++j)
      cout << " ";
    DescriptionAttribute* description = properties[i]->getAttribute<DescriptionAttribute>();
    if (description)
      cout << "  " << description->getDescription();
    cout << endl;
  }
  cout << endl;
}

void ArgumentsParser::PrintDefaultUsage(const std::string& programName, const reflection::ReflectableClass& object, bool showOnlyParameters) {
  ArgumentsParser::PrintUsage(string("\nUsage: ") + programName + " [switches], where switches are:", object, showOnlyParameters);
}

}
