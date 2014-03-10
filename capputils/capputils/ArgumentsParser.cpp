/*
 * ArgumentsParser.cpp
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#include <capputils/ArgumentsParser.h>

#include <cstring>
#include <iostream>
#include <cmath>

#include <capputils/attributes/FlagAttribute.h>
#include <capputils/attributes/DescriptionAttribute.h>
#include <capputils/attributes/HideAttribute.h>
#include <capputils/attributes/ParameterAttribute.h>
#include <capputils/attributes/OperandAttribute.h>
#include <capputils/attributes/EnumerableAttribute.h>
#include <capputils/attributes/EnumeratorAttribute.h>

#include <stdexcept>

namespace capputils {

using namespace reflection;
using namespace attributes;

ParameterDescription::ParameterDescription() : object(NULL), property(NULL) { }

ParameterDescription::ParameterDescription(ReflectableClass* object, IClassProperty* property, const std::string& longName,
    const std::string& shortName, const std::string& description)
: object(object), property(property), longName(longName), shortName(shortName), description(description) { }

boost::shared_ptr<std::vector<std::string> > ArgumentsParser::Parse(ReflectableClass& object, int argc, char** argv, bool parseOnlyParameter) {
  ParameterDescriptions parameters;
  CreateParameterDescriptions(object, parseOnlyParameter, parameters);
  return Parse(parameters, argc, argv, parseOnlyParameter);
}

boost::shared_ptr<std::vector<std::string> > ArgumentsParser::Parse(ParameterDescriptions& parameters, int argc, char** argv, bool parseOnlyParameter) {
  boost::shared_ptr<std::vector<std::string> > unhandledArguments(new std::vector<std::string>());
  bool operandParsed = false;

  std::map<std::string, ParameterDescription> parameterMap;
  for (size_t i = 0; i < parameters.parameters.size(); ++i) {
    if (parameters.parameters[i].shortName.size())
      parameterMap[std::string("-") + parameters.parameters[i].shortName] = parameters.parameters[i];
    if (parameters.parameters[i].longName.size())
      parameterMap[std::string("--") + parameters.parameters[i].longName] = parameters.parameters[i];
  }

  IEnumerableAttribute* enumerable = NULL;
  boost::shared_ptr<reflection::IPropertyIterator> operandIterator;

  if (parameters.operands && (enumerable = parameters.operands->property->getAttribute<IEnumerableAttribute>())) {
    enumerable->renewCollection(*parameters.operands->object, parameters.operands->property);
    operandIterator = enumerable->getPropertyIterator(*parameters.operands->object, parameters.operands->property);
  }

  for (int iArg = 1; iArg < argc; ++iArg) {

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
      // Parse operands
      if (operandIterator) {
        operandIterator->setStringValue(*parameters.operands->object, argv[iArg]);
        operandIterator->next();
      } else if (parameters.operands && !operandParsed) {
        parameters.operands->property->setStringValue(*parameters.operands->object, argv[iArg]);
        operandParsed = true;
      } else {
        unhandledArguments->push_back(argv[iArg]);
      }
    }
  }
  return unhandledArguments;
}

void ArgumentsParser::PrintUsage(const std::string& header, ReflectableClass& object, bool showOnlyParameters) {
  ParameterDescriptions parameters;
  CreateParameterDescriptions(object, showOnlyParameters, parameters);
  PrintUsage(header, parameters);
}

void ArgumentsParser::PrintUsage(const std::string& header, ParameterDescriptions& parameters) {
  attributes::IEnumeratorAttribute* enumAttribute = NULL;

  int column1Width = 0, column2Width = 0, column3Width = 0;
  for (size_t i = 0; i < parameters.parameters.size(); ++i) {
    column1Width = std::max(column1Width, (int)parameters.parameters[i].shortName.size());
    column2Width = std::max(column2Width, (int)parameters.parameters[i].longName.size());
  }

  std::cout << header << std::endl << std::endl;

  if (parameters.operands && (parameters.operands->description.size() || parameters.operands->property->getAttribute<attributes::IEnumeratorAttribute>())) {

    column3Width = std::max(0, (int)parameters.operands->longName.size() + 2 - (column1Width ? column1Width + 2 : 0) - (column2Width ? column2Width + 3 : 0) + 1);

    std::cout << "  <" << parameters.operands->longName << "> ";
    for (int j = 0; j < (column1Width ? column1Width + 2 : 0) + (column2Width ? column2Width + 3 : 0) + column3Width - (int)parameters.operands->longName.size() - 2; ++j)
      std::cout << " ";

    std::cout << parameters.operands->description;

    if ((enumAttribute = parameters.operands->property->getAttribute<attributes::IEnumeratorAttribute>())) {
      boost::shared_ptr<capputils::AbstractEnumerator> enumerator = enumAttribute->getEnumerator(*parameters.operands->object, parameters.operands->property);
      if (enumerator) {
        if (parameters.operands->description.size())
          std::cout << " (";
        else
          std::cout << "Possible values are: ";
        std::vector<std::string>& values = enumerator->getValues();
        for (size_t iVal = 0; iVal < values.size(); ++iVal) {
          if (iVal)
            std::cout << ", ";
          std::cout << values[iVal];
        }
        if (parameters.operands->description.size())
          std::cout << ")";
      }
    }

    std::cout << "\n" << std::endl;
  }

  for (size_t i = 0; i < parameters.parameters.size(); ++i) {
    std::string shortName = parameters.parameters[i].shortName;
    std::string fullName = parameters.parameters[i].longName;
    std::cout << " ";
    if (shortName.size())
      std::cout << " -" << shortName;
    else if (column1Width)
      std::cout << "  ";
    for (int j = 0; j < column1Width - (int)shortName.size(); ++j)
      std::cout << " ";
    if (fullName.size())
      std::cout << " --" << fullName;
    else if (column2Width)
      std::cout << "   ";
    for (int j = 0; j < column2Width - (int)fullName.size(); ++j)
      std::cout << " ";
    for (int j = 0; j < column3Width; ++j)
      std::cout << " ";
    std::cout << "  " << parameters.parameters[i].description;

    if ((enumAttribute = parameters.parameters[i].property->getAttribute<attributes::IEnumeratorAttribute>())) {
      boost::shared_ptr<capputils::AbstractEnumerator> enumerator = enumAttribute->getEnumerator(*parameters.parameters[i].object, parameters.parameters[i].property);
      if (enumerator) {
        if (parameters.parameters[i].description.size())
          std::cout << " (";
        else
          std::cout << "Possible values are: ";
        std::vector<std::string>& values = enumerator->getValues();
        for (size_t iVal = 0; iVal < values.size(); ++iVal) {
          if (iVal)
            std::cout << ", ";
          std::cout << values[iVal];
        }
        if (parameters.parameters[i].description.size())
          std::cout << ")";
      }
    }

    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void ArgumentsParser::PrintDefaultUsage(const std::string& programName, ReflectableClass& object, bool showOnlyParameters) {
  ParameterDescriptions parameters;
  CreateParameterDescriptions(object, showOnlyParameters, parameters);
  ArgumentsParser::PrintDefaultUsage(programName, parameters);
}

void ArgumentsParser::PrintDefaultUsage(const std::string& programName, ParameterDescriptions& parameters) {
  if (parameters.operands)
    ArgumentsParser::PrintUsage(std::string("\nUsage: ") + programName + " <" + parameters.operands->longName + "> [switches], where switches are:", parameters);
  else
    ArgumentsParser::PrintUsage(std::string("\nUsage: ") + programName + " [switches], where switches are:", parameters);
}

void ArgumentsParser::CreateParameterDescriptions(ReflectableClass& object, bool includeOnlyParameters, ParameterDescriptions& parameters) {
  std::vector<IClassProperty*>& properties = object.getProperties();
  ParameterAttribute* parameter = NULL;
  OperandAttribute* operand = NULL;
  DescriptionAttribute* description = NULL;

  for (size_t i = 0; i < properties.size(); ++i) {
    parameter = properties[i]->getAttribute<ParameterAttribute>();
    description = properties[i]->getAttribute<DescriptionAttribute>();
    operand = properties[i]->getAttribute<OperandAttribute>();

    if (properties[i]->getAttribute<HideAttribute>() || (includeOnlyParameters && !parameter))
      continue;

    if (operand) {
      if (parameters.operands)
        throw std::logic_error("Multiple operands detected. Only one property can be set to handle operands.");
      parameters.operands = boost::make_shared<ParameterDescription>(&object, properties[i], parameter->getLongName(), parameter->getShortName(), (description ? description->getDescription() : ""));
    } else if (parameter) {
      parameters.parameters.push_back(ParameterDescription(&object, properties[i], parameter->getLongName(), parameter->getShortName(), (description ? description->getDescription() : "")));
    } else {
      parameters.parameters.push_back(ParameterDescription(&object, properties[i], properties[i]->getName(), "", (description ? description->getDescription() : "")));
    }
  }
}

}
