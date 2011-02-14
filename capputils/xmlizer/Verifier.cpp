/*
 * Verifier.cpp
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#include "Verifier.h"

#include "IAssertionAttribute.h"

using namespace reflection;
using namespace std;

namespace xmlizer {

using namespace attributes;

Verifier::Verifier() {
  // TODO Auto-generated constructor stub

}

Verifier::~Verifier() {
  // TODO Auto-generated destructor stub
}

bool Verifier::Valid(const ReflectableClass& object, const ClassProperty& property, ostream& stream) {
  bool isValid = true;

  vector<IAttribute*> attributes = property.attributes;
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

  vector<ClassProperty*>& properties = object.getProperties();
  for (unsigned i = 0; i < properties.size(); ++i)
    if (!Verifier::Valid(object, *properties[i], stream))
      isValid = false;

  return isValid;
}

}
