/*
 * Verifier.h
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#ifndef VERIFIER_H_
#define VERIFIER_H_

#include <capputils/reflection/ReflectableClass.h>

#include <ostream>
#include <iostream>

namespace capputils {

class Logbook;

class Verifier {
public:
  Verifier();
  virtual ~Verifier();

  static bool Valid(const reflection::ReflectableClass& object,
      const reflection::IClassProperty& property, std::ostream& stream = std::cout);

  static bool Valid(const reflection::ReflectableClass& object,
      const reflection::IClassProperty& property, Logbook& logbook);

  static bool Valid(const reflection::ReflectableClass& object,
      std::ostream& stream = std::cout);

  static bool Valid(const reflection::ReflectableClass& object, Logbook& logbook);

  static bool UpToDate(const reflection::ReflectableClass& object);
};

}

#endif /* VERIFIER_H_ */
