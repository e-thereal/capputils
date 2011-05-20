/*
 * Verifier.h
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#ifndef VERIFIER_H_
#define VERIFIER_H_

#include "ReflectableClass.h"

#include <ostream>
#include <iostream>

namespace capputils {

class Verifier {
public:
  Verifier();
  virtual ~Verifier();

  static bool Valid(const reflection::ReflectableClass& object,
      const reflection::IClassProperty& property, std::ostream& stream = std::cout);
  static bool Valid(const reflection::ReflectableClass& object,
      std::ostream& stream = std::cout);
  static bool UpToDate(const reflection::ReflectableClass& object);
};

}

#endif /* VERIFIER_H_ */
