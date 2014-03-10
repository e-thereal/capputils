/*
 * GenerateBashCompletion.cpp
 *
 *  Created on: Oct 19, 2012
 *      Author: tombr
 */

#include <capputils/GenerateBashCompletion.h>

#include <capputils/attributes/FlagAttribute.h>
#include <capputils/attributes/FilenameAttribute.h>
#include <capputils/attributes/EnumeratorAttribute.h>
#include <capputils/attributes/ParameterAttribute.h>
#include <capputils/attributes/OperandAttribute.h>
#include <capputils/attributes/HideAttribute.h>

#include <fstream>

using namespace capputils::attributes;

namespace capputils {

void addBashCompletionEntry(const std::string& parameterName, const reflection::ReflectableClass& object, reflection::IClassProperty* property, std::ostream& out) {
  FilenameAttribute* filename = NULL;
  IEnumeratorAttribute* enumAttr = NULL;

  if (parameterName.size()) {
    out <<
"        " << parameterName << ")\n";
  }
  if ((filename = property->getAttribute<FilenameAttribute>())) {
    if (filename->getPattern().size())
      out <<
"          files=( $(compgen -f -X \"!" << filename->getMatchPattern() << "\" ${cur}) )\n";
    else
      out <<
"          files=( $(compgen -f ${cur}) )\n";
    out <<
"          directories=( $(compgen -d -S / ${cur}) )\n"
"          COMPREPLY=( \"${files[@]/%/ }\" \"${directories[@]}\" )\n";
  } else if ((enumAttr = property->getAttribute<IEnumeratorAttribute>())) {
    boost::shared_ptr<AbstractEnumerator> enumerator = enumAttr->getEnumerator(object, property);
    std::vector<std::string>& values = enumerator->getValues();
    out <<
"          options=( $(compgen -W \"";
    for (size_t j = 0; j < values.size(); ++j) {
      if (j > 0)
        out << " ";
      out << values[j];
    }
    out << "\" -- ${cur}) )\n"
"          COMPREPLY=( \"${options[@]/%/ }\" )\n";
  } else {
    out <<
"          COMPREPLY=()\n";
  }
  out <<
"          return 0\n";
  if (parameterName.size()) {
    out <<
"          ;;\n";
  }
}

void GenerateBashCompletion::Generate(const std::string& programName, const reflection::ReflectableClass& object, std::ostream& out, bool parseOnlyParameters) {
  ParameterAttribute* parameter = NULL;
  OperandAttribute* operand = NULL;
  bool addWhiteSpace = false;

  reflection::IClassProperty* operandProperty = NULL;

  std::vector<reflection::IClassProperty*>& properties = object.getProperties();
  for (size_t i = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<OperandAttribute>()) {
      operandProperty = properties[i];
      break;
    }
  }

  out << "_" << programName << "()\n"
"  {\n"
"      local cur prev opts\n"
"      COMPREPLY=()\n"
"      cur=\"${COMP_WORDS[COMP_CWORD]}\"\n"
"      prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n"
"      opts=\"";

  for (size_t i = 0; i < properties.size(); ++i) {
    parameter = properties[i]->getAttribute<ParameterAttribute>();
    operand = properties[i]->getAttribute<OperandAttribute>();

    if (properties[i]->getAttribute<HideAttribute>() || (parseOnlyParameters && !parameter) || operand)
      continue;

    if (addWhiteSpace)
      out << " ";

    if (parameter) {
      if (parameter->getShortName().size())
        out << "-" << parameter->getShortName();

      if (parameter->getShortName().size() && parameter->getLongName().size())
        out << " ";

      if (parameter->getLongName().size())
        out << "--" << parameter->getLongName();
    } else {
      out << "--" << properties[i]->getName();
    }

    addWhiteSpace = true;
  }
  out << "\"\n"
"      \n"
"      case \"${prev}\" in\n";
  for (size_t i = 0; i < properties.size(); ++i) {
    parameter = properties[i]->getAttribute<ParameterAttribute>();
    operand = properties[i]->getAttribute<OperandAttribute>();

    if (properties[i]->getAttribute<HideAttribute>() || (parseOnlyParameters && !parameter) || operand)
      continue;

    if (!properties[i]->getAttribute<FlagAttribute>()) {
      if (parameter) {
        if (parameter->getLongName().size()) {
          addBashCompletionEntry(std::string("--") + parameter->getLongName(), object, properties[i], out);
        }

        if (parameter->getShortName().size()) {
          addBashCompletionEntry(std::string("-") + parameter->getShortName(), object, properties[i], out);
        }
      } else {
        addBashCompletionEntry(std::string("--") + properties[i]->getName(), object, properties[i], out);
      }
    }
  }
  out <<
"        *)\n"
"          ;;\n"
"      esac\n"
"      \n";
  if (operandProperty) {
    out <<
"      if [[ ${cur} == -* ]] ; then\n"
"          options=( $(compgen -W \"${opts}\" -- ${cur}) )\n"
"          COMPREPLY=( \"${options[@]/%/ }\" )\n"
"          return 0\n"
"      else\n";
    addBashCompletionEntry("", object, operandProperty, out);
    out <<
"      fi\n";
  } else {
    out <<
"      options=( $(compgen -W \"${opts}\" -- ${cur}) )\n"
"      COMPREPLY=( \"${options[@]/%/ }\" )\n"
"      return 0\n";
  }
  out <<
"  }\n"
"  complete -o nospace -F _" << programName << " " << programName << std::endl;
}

void GenerateBashCompletion::Generate(const std::string& programName, const reflection::ReflectableClass& object, const std::string& filename, bool parseOnlyParameters) {
  std::ofstream out(filename.c_str());
  GenerateBashCompletion::Generate(programName, object, out, parseOnlyParameters);
  out.close();
}

} /* namespace capputils */
