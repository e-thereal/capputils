/*
 * Verifier.cpp
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#include "Verifier.h"

#include "IAssertionAttribute.h"

using namespace std;
using namespace capputils::reflection;
using namespace capputils::attributes;

namespace capputils {

Verifier::Verifier() {
}

Verifier::~Verifier() {
}

bool Verifier::Valid(const ReflectableClass& object, const IClassProperty& property, ostream& stream) {
  bool isValid = true;

  vector<IAttribute*> attributes = property.getAttributes();
  for (unsigned j = 0; j < attributes.size(); ++j) {
    IAssertionAttribute* assertion = dynamic_cast<IAssertionAttribute*>(attributes[j]);
    if (assertion) {
      if (!assertion->valid(property, object)) {
        isValid = false;
        stream << assertion->getLastMessage() << endl;
      }
    }
  }

  return isValid;
}

bool Verifier::Valid(const ReflectableClass& object, ostream& stream) {
  bool isValid = true;

  vector<IClassProperty*>& properties = object.getProperties();
  for (unsigned i = 0; i < properties.size(); ++i)
    if (!Verifier::Valid(object, *properties[i], stream))
      isValid = false;

  return isValid;
}

}
