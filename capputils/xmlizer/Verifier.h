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

namespace xmlizer {

class Verifier {
public:
  Verifier();
  virtual ~Verifier();

  static bool Valid(const reflection::ReflectableClass& object,
      const reflection::ClassProperty& property, std::ostream& stream);
  static bool Valid(const reflection::ReflectableClass& object,
      std::ostream& stream = std::cout);
};

}

#endif /* VERIFIER_H_ */
