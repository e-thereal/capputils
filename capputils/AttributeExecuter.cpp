/*
 * AttributeExecuter.cpp
 *
 *  Created on: Mar 2, 2011
 *      Author: tombr
 */

#include <capputils/AttributeExecuter.h>

#include <capputils/attributes/IExecutableAttribute.h>

using namespace capputils::reflection;
using namespace capputils::attributes;
using namespace std;

namespace capputils {

void AttributeExecuter::ExecuteBefore(ReflectableClass& object, const IClassProperty& property) {
  const vector<IAttribute*>& attributes = property.getAttributes();
  for (unsigned j = 0; j < attributes.size(); ++j) {
    IExecutableAttribute* executable = dynamic_cast<IExecutableAttribute*>(attributes[j]);
    if (executable) {
      executable->executeBefore(object, property);
    }
  }
}

void AttributeExecuter::ExecuteAfter(ReflectableClass& object, const IClassProperty& property) {
  const vector<IAttribute*>& attributes = property.getAttributes();
  for (unsigned j = 0; j < attributes.size(); ++j) {
    IExecutableAttribute* executable = dynamic_cast<IExecutableAttribute*>(attributes[j]);
    if (executable) {
      executable->executeAfter(object, property);
    }
  }
}

}
