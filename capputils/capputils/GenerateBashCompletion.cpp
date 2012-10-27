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

#include <fstream>

using namespace capputils::attributes;

namespace capputils {

void GenerateBashCompletion::Generate(const std::string& programName, const reflection::ReflectableClass& object, std::ostream& out) {
  out << "_" << programName << "()\n"
"  {\n"
"      local cur prev opts\n"
"      COMPREPLY=()\n"
"      cur=\"${COMP_WORDS[COMP_CWORD]}\"\n"
"      prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n"
"      opts=\"";
  std::vector<reflection::IClassProperty*>& properties = object.getProperties();
  for (size_t i = 0; i < properties.size(); ++i) {
    if (i > 0)
      out << " ";
    out << "--" << properties[i]->getName();
  }
  out << "\"\n"
"      \n"
"      case \"${prev}\" in\n";
  for (size_t i = 0; i < properties.size(); ++i) {
    if (!properties[i]->getAttribute<FlagAttribute>()) {
      IEnumeratorAttribute* enumAttr;
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

void GenerateBashCompletion::Generate(const std::string& programName, const reflection::ReflectableClass& object, const std::string& filename) {
  std::ofstream out(filename.c_str());
  GenerateBashCompletion::Generate(programName, object, out);
  out.close();
}

} /* namespace capputils */
