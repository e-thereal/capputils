/*
 * GenerateBashCompletion.cpp
 *
 *  Created on: Oct 19, 2012
 *      Author: tombr
 */

#include "GenerateBashCompletion.h"

#include <capputils/FlagAttribute.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/EnumeratorAttribute.h>
#include <capputils/ParameterAttribute.h>
#include <capputils/HideAttribute.h>

#include <fstream>

using namespace capputils::attributes;

namespace capputils {

void GenerateBashCompletion::Generate(const std::string& programName, const reflection::ReflectableClass& object, std::ostream& out, bool parseOnlyParameters) {
  ParameterAttribute* parameter = NULL;
  bool addWhiteSpace = false;

  out << "_" << programName << "()\n"
"  {\n"
"      local cur prev opts\n"
"      COMPREPLY=()\n"
"      cur=\"${COMP_WORDS[COMP_CWORD]}\"\n"
"      prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n"
"      opts=\"";
  std::vector<reflection::IClassProperty*>& properties = object.getProperties();
  for (size_t i = 0; i < properties.size(); ++i) {
    parameter = properties[i]->getAttribute<ParameterAttribute>();

    if (properties[i]->getAttribute<HideAttribute>() || (parseOnlyParameters && !parameter))
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

    if (properties[i]->getAttribute<HideAttribute>() || (parseOnlyParameters && !parameter))
      continue;

    if (!properties[i]->getAttribute<FlagAttribute>()) {
      IEnumeratorAttribute* enumAttr;

      if (parameter) {
        if (parameter->getLongName().size()) {
          out <<
    "        --" << parameter->getLongName() << ")\n";
          if (properties[i]->getAttribute<FilenameAttribute>()) {
            out <<
    "          COMPREPLY=( $(compgen -f ${cur}) )\n";
          } else if ((enumAttr = properties[i]->getAttribute<IEnumeratorAttribute>())) {
            boost::shared_ptr<AbstractEnumerator> enumerator = enumAttr->getEnumerator(object, properties[i]);
            std::vector<std::string>& values = enumerator->getValues();
            out <<
    "          COMPREPLY=( $(compgen -W \"";
            for (size_t j = 0; j < values.size(); ++j) {
              if (j > 0)
                out << " ";
              out << values[j];
            }
            out << "\" -- ${cur}) )\n";
          } else {
            out <<
    "          COMPREPLY=()\n";
          }
          out <<
    "          return 0\n"
    "          ;;\n";
        }

        if (parameter->getShortName().size()) {
          out <<
    "        -" << parameter->getShortName() << ")\n";
          if (properties[i]->getAttribute<FilenameAttribute>()) {
            out <<
    "          COMPREPLY=( $(compgen -f ${cur}) )\n";
          } else if ((enumAttr = properties[i]->getAttribute<IEnumeratorAttribute>())) {
            boost::shared_ptr<AbstractEnumerator> enumerator = enumAttr->getEnumerator(object, properties[i]);
            std::vector<std::string>& values = enumerator->getValues();
            out <<
    "          COMPREPLY=( $(compgen -W \"";
            for (size_t j = 0; j < values.size(); ++j) {
              if (j > 0)
                out << " ";
              out << values[j];
            }
            out << "\" -- ${cur}) )\n";
          } else {
            out <<
    "          COMPREPLY=()\n";
          }
          out <<
    "          return 0\n"
    "          ;;\n";
        }
      } else {

        out <<
  "        --" << properties[i]->getName() << ")\n";
        if (properties[i]->getAttribute<FilenameAttribute>()) {
          out <<
  "          COMPREPLY=( $(compgen -f ${cur}) )\n";
        } else if ((enumAttr = properties[i]->getAttribute<IEnumeratorAttribute>())) {
          boost::shared_ptr<AbstractEnumerator> enumerator = enumAttr->getEnumerator(object, properties[i]);
          std::vector<std::string>& values = enumerator->getValues();
          out <<
  "          COMPREPLY=( $(compgen -W \"";
          for (size_t j = 0; j < values.size(); ++j) {
            if (j > 0)
              out << " ";
            out << values[j];
          }
          out << "\" -- ${cur}) )\n";
        } else {
          out <<
  "          COMPREPLY=()\n";
        }
        out <<
  "          return 0\n"
  "          ;;\n";
      }
    }
  }
  out <<
"        *)\n"
"          ;;\n"
"      esac\n"
"      \n"
"  #    if [[ ${cur} == -* ]] ; then\n"
"          COMPREPLY=( $(compgen -W \"${opts}\" -- ${cur}) )\n"
"          return 0\n"
"  #    fi\n"
"  }\n"
"  complete -F _" << programName << " " << programName << std::endl;
}

void GenerateBashCompletion::Generate(const std::string& programName, const reflection::ReflectableClass& object, const std::string& filename, bool parseOnlyParameters) {
  std::ofstream out(filename.c_str());
  GenerateBashCompletion::Generate(programName, object, out, parseOnlyParameters);
  out.close();
}

} /* namespace capputils */
