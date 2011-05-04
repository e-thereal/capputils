/*
 * Student.cpp
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#include "Student.h"

BeginPropertyDefinitions(Student)

  ReflectableBase(Person)

  DefineProperty(Matrikel)

EndPropertyDefinitions

Student::Student() : _Matrikel(0) {
}

Student::~Student() {
}
