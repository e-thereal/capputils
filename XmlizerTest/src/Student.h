/*
 * Student.h
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#ifndef STUDENT_H_
#define STUDENT_H_

#include "Person.h"

class Student : public Person {

  InitReflectableClass(Student)

  Property(Matrikel, int)

public:
  Student();
  virtual ~Student();
};

#endif /* STUDENT_H_ */
