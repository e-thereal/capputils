/*
 * AttributeExecuter.cpp
 *
 *  Created on: Mar 2, 2011
 *      Author: tombr
 */

#include "AttributeExecuter.h"

#include "IExecutableAttribute.h"

using namespace capputils::reflection;
using namespace std;

namespace capputils {

namespace attributes {

AttributeExecuter::AttributeExecuter() {
}

AttributeExecuter::~AttributeExecuter() {
}

void AttributeExecuter::Execute(ReflectableClass& object, const IClassProperty& property) {
  vector<IAttribute*> attributes = property.getAttributes();
  for (unsigned j = 0; j < attributes.size(); ++j) {
    IExecutableAttribute* executable = dynamic_cast<IExecutableAttribute*>(attributes[j]);
    if (executable) {
      executable->execute(object);
    }
  }
}

}

}
