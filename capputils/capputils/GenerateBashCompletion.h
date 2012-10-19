/*
 * GenerateBashCompletion.h
 *
 *  Created on: Oct 19, 2012
 *      Author: tombr
 */

#ifndef CAPPUTILS_GENERATEBASHCOMPLETION_H_
#define CAPPUTILS_GENERATEBASHCOMPLETION_H_

#include <capputils/ReflectableClass.h>
#include <iostream>

namespace capputils {

class GenerateBashCompletion {
public:
  static void Generate(const std::string& programName, const reflection::ReflectableClass& object, std::ostream& out);
  static void Generate(const std::string& programName, const reflection::ReflectableClass& object, const std::string& filename);
};

} /* namespace capputils */

#endif /* CAPPUTILS_GENERATEBASHCOMPLETION_H_ */
