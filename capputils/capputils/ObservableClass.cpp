/*
 * ObservableClass.cpp
 *
 *  Created on: Mar 2, 2011
 *      Author: tombr
 */

#include "ObservableClass.h"

#include <iostream>

using namespace std;

namespace capputils {

ObservableClass::ObservableClass() {
}

ObservableClass::~ObservableClass() {
  while (!parents.empty()) {
    parents.begin()->first->removeChild(this);
  }
  for (set<ObservableClass*>::iterator i = children.begin(); i != children.end(); ++i)
    (*i)->removeParent(this);
}

void ObservableClass::fireEvent(int eventId) {
  Changed(this, eventId);
  for (map<ObservableClass*, int>::iterator i = parents.begin(); i != parents.end(); ++i)
    i->first->fireEvent(i->second);
}

void ObservableClass::addChild(ObservableClass* child, int eventId) {
  if (children.find(child) == children.end()) {
    children.insert(child);
    child->addParent(this, eventId);
  } else {
    throw "Adding a child twice is not allowed";
  }
}

void ObservableClass::removeChild(ObservableClass* child) {
  if (children.find(child) != children.end()) {
    children.erase(child);
    child->removeParent(this);
  }
}

void ObservableClass::addParent(ObservableClass* parent, int eventId) {
  parents[parent] = eventId;
}

void ObservableClass::removeParent(ObservableClass* parent) {
  if (parents.find(parent) != parents.end()) {
    parents.erase(parent);
  }
}

}
