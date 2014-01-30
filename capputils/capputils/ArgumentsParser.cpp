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
#include "EnumerableAttribute.h"
#include "EnumeratorAttribute.h"

namespace capputils {

using namespace reflection;
using namespace attributes;

ParameterDescription::ParameterDescription() : object(NULL), property(NULL) { }

ParameterDescription::ParameterDescription(ReflectableClass* object, IClassProperty* property, const std::string& longName,
    const std::string& shortName, const std::string& description)
: object(object), property(property), longName(longName), shortName(shortName), description(description) { }

void ArgumentsParser::Parse(ReflectableClass& object, int argc, char** argv, bool parseOnlyParameter) {
  std::vector<ParameterDescription> parameters;
  CreateParameterList(object, parseOnlyParameter, parameters);
  Parse(parameters, argc, argv, parseOnlyParameter);
}

void ArgumentsParser::Parse(std::vector<ParameterDescription>& parameters, int argc, char** argv, bool parseOnlyParameter) {
  std::map<std::string, ParameterDescription> parameterMap;
  for (size_t i = 0; i < parameters.size(); ++i) {
    if (parameters[i].shortName.size())
      parameterMap[std::string("-") + parameters[i].shortName] = parameters[i];
    if (parameters[i].longName.size())
      parameterMap[std::string("--") + parameters[i].longName] = parameters[i];
  }

  IEnumerableAttribute* enumerable = NULL;

  for (int iArg = 0; iArg < argc; ++iArg) {

    if (parameterMap.find(argv[iArg]) != parameterMap.end()) {
      ParameterDescription parameter = parameterMap[argv[iArg]];
      if (parameter.property->getAttribute<FlagAttribute>()) {
        if (iArg + 1 >= argc || parameterMap.find(argv[iArg + 1]) != parameterMap.end())
          parameter.property->setStringValue(*parameter.object, "1");
        else
          parameter.property->setStringValue(*parameter.object, argv[++iArg]);
      } else if ((enumerable = parameter.property->getAttribute<IEnumerableAttribute>())) {
        enumerable->renewCollection(*parameter.object, parameter.property);
        boost::shared_ptr<reflection::IPropertyIterator> iter = enumerable->getPropertyIterator(*parameter.object, parameter.property);
        for (iter->reset(); iArg < argc - 1 && parameterMap.find(argv[iArg + 1]) == parameterMap.end(); iter->next())
          iter->setStringValue(*parameter.object, argv[++iArg]);
      } else if (iArg < argc - 1) {
        parameter.property->setStringValue(*parameter.object, argv[++iArg]);
      }
    } else {
      // default parameter
    }
  }
}

void ArgumentsParser::PrintUsage(const std::string& header, ReflectableClass& object, bool showOnlyParameters) {
  std::vector<ParameterDescription> parameters;
  CreateParameterList(object, showOnlyParameters, parameters);
  PrintUsage(header, parameters);
}

void ArgumentsParser::PrintUsage(const std::string& header, std::vector<ParameterDescription>& parameters) {
  attributes::IEnumeratorAttribute* enumAttribute = NULL;

  size_t column1Width = 0, column2Width = 0;
  for (size_t i = 0; i < parameters.size(); ++i) {
    column1Width = std::max(column1Width, parameters[i].shortName.size());
    column2Width = std::max(column2Width, parameters[i].longName.size());
  }

  std::cout << header << std::endl << std::endl;
  for (size_t i = 0; i < parameters.size(); ++i) {
    std::string shortName = parameters[i].shortName;
    std::string fullName = parameters[i].longName;
    std::cout << " ";
    if (shortName.size())
      std::cout << " -" << shortName;
    else if (column1Width)
      std::cout << "  ";
    for (size_t j = 0; j < column1Width - shortName.size(); ++j)
      std::cout << " ";
    if (fullName.size())
      std::cout << " --" << fullName;
    else if (column2Width)
      std::cout << "   ";
    for (size_t j = 0; j < column2Width - fullName.size(); ++j)
      std::cout << " ";
    std::cout << "  " << parameters[i].description;

    if ((enumAttribute = parameters[i].property->getAttribute<attributes::IEnumeratorAttribute>())) {
      boost::shared_ptr<capputils::AbstractEnumerator> enumerator = enumAttribute->getEnumerator(*parameters[i].object, parameters[i].property);
      if (enumerator) {
        if (parameters[i].description.size())
          std::cout << " (";
        else
          std::cout << "Possible values are: ";
        std::vector<std::string>& values = enumerator->getValues();
        for (size_t iVal = 0; iVal < values.size(); ++iVal) {
          if (iVal)
            std::cout << ", ";
          std::cout << values[iVal];
        }
        if (parameters[i].description.size())
          std::cout << ")";
      }
    }

    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void ArgumentsParser::PrintDefaultUsage(const std::string& programName, ReflectableClass& object, bool showOnlyParameters) {
  ArgumentsParser::PrintUsage(std::string("\nUsage: ") + programName + " [switches], where switches are:", object, showOnlyParameters);
}

void ArgumentsParser::PrintDefaultUsage(const std::string& programName, std::vector<ParameterDescription>& parameters) {
  ArgumentsParser::PrintUsage(std::string("\nUsage: ") + programName + " [switches], where switches are:", parameters);
}

void ArgumentsParser::CreateParameterList(ReflectableClass& object, bool includeOnlyParameters, std::vector<ParameterDescription>& parameters) {
  std::vector<IClassProperty*>& properties = object.getProperties();
  ParameterAttribute* parameter = NULL;
  DescriptionAttribute* description = NULL;

  for (size_t i = 0; i < properties.size(); ++i) {
    parameter = properties[i]->getAttribute<ParameterAttribute>();
    description = properties[i]->getAttribute<DescriptionAttribute>();

    if (properties[i]->getAttribute<HideAttribute>() || (includeOnlyParameters && !parameter))
      continue;

    if (parameter)
      parameters.push_back(ParameterDescription(&object, properties[i], parameter->getLongName(), parameter->getShortName(), (description ? description->getDescription() : "")));
    else
      parameters.push_back(ParameterDescription(&object, properties[i], properties[i]->getName(), "", (description ? description->getDescription() : "")));
  }
}

}
