/*
 * Verifier.cpp
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#include <capputils/Verifier.h>

#include <capputils/attributes/IAssertionAttribute.h>
#include <capputils/attributes/FilenameAttribute.h>
#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <capputils/attributes/TimeStampAttribute.h>


#include <cmath>
#include <iostream>

#include <capputils/Logbook.h>

using namespace std;
using namespace capputils::reflection;
using namespace capputils::attributes;
using namespace boost::filesystem;

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

bool Verifier::Valid(const ReflectableClass& object, const IClassProperty& property, Logbook& logbook) {
  bool isValid = true;

  vector<IAttribute*> attributes = property.getAttributes();
  for (unsigned j = 0; j < attributes.size(); ++j) {
    IAssertionAttribute* assertion = dynamic_cast<IAssertionAttribute*>(attributes[j]);
    if (assertion) {
      if (!assertion->valid(property, object)) {
        isValid = false;
        logbook(Severity::Warning) << assertion->getLastMessage();
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

bool Verifier::Valid(const reflection::ReflectableClass& object, Logbook& logbook) {
  bool isValid = true;

  vector<IClassProperty*>& properties = object.getProperties();
  for (unsigned i = 0; i < properties.size(); ++i)
    if (!Verifier::Valid(object, *properties[i], logbook))
      isValid = false;

  return isValid;
}

bool Verifier::UpToDate(const ReflectableClass& object) {
  vector<IClassProperty*>& properties = object.getProperties();
  FilenameAttribute* fa;
  time_t newestInput = 0, oldestOutput = 0;
  bool hasOutputDate = false;
  int newestInputId = -1;
  int oldestOutputId = -1;

  try {
    for (unsigned i = 0; i < properties.size(); ++i) {
      IClassProperty* prop = properties[i];
      TimeStampAttribute* timeStamp = prop->getAttribute<TimeStampAttribute>();

      if (prop->getAttribute<OutputAttribute>()) {
        time_t newOutputDate;

        if ((fa = prop->getAttribute<FilenameAttribute>()) && !fa->getMultipleSelection()) {
          newOutputDate = last_write_time(prop->getStringValue(object));
        } else if (timeStamp) {
          newOutputDate = timeStamp->getTime(object);
        } else {
          continue;
        }

        if (!hasOutputDate) {
          oldestOutput = newOutputDate;
          hasOutputDate = true;
          oldestOutputId = i;
        }
        if (newOutputDate < oldestOutput)
          oldestOutputId = i;
        oldestOutput = min(oldestOutput, newOutputDate);
      }
      // Inputs are attributes that are not pure outputs.
      if (prop->getAttribute<OutputAttribute>() == 0 || prop->getAttribute<InputAttribute>() != 0){
        if ((fa = prop->getAttribute<FilenameAttribute>()) && !fa->getMultipleSelection() && prop->getAttribute<InputAttribute>()) {
          if (last_write_time(prop->getStringValue(object)) > newestInput)
            newestInputId = i;
          newestInput = max(newestInput, last_write_time(prop->getStringValue(object)));
        } else if (timeStamp) {
          if (timeStamp->getTime(object) > newestInput)
            newestInputId = i;
          newestInput = max(newestInput, timeStamp->getTime(object));
        }
      }
    }
  } catch (filesystem_error error) {
    return false;
  }
//  if (newestInput <= oldestOutput) {
//    cout << object.getClassName() << " already up to date!" << endl;
//    if (newestInputId > -1)
//      cout << "  Newest Input: " << properties[newestInputId]->getName() << endl;
//    if (oldestOutputId > -1)
//      cout << "  Oldest Output: " << properties[oldestOutputId]->getName() << endl;
//  } else {
//    cout << object.getClassName() << " is not up to date!" << endl;
//    if (newestInputId > -1)
//      cout << "  Newest Input: " << properties[newestInputId]->getName() << endl;
//    if (oldestOutputId > -1)
//      cout << "  Oldest Output: " << properties[oldestOutputId]->getName() << endl;
//  }
  return (newestInput <= oldestOutput);
}

}
