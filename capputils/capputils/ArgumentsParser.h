/*
 * ArgumentsParser.h
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#ifndef ARGUMENTSPARSER_H_
#define ARGUMENTSPARSER_H_

#include "capputils.h"
#include "ReflectableClass.h"

namespace capputils {

class CAPPUTILS_API ArgumentsParser {
public:
  ArgumentsParser();
  virtual ~ArgumentsParser();

  static void Parse(reflection::ReflectableClass& object, int argc, char** argv);
  static void PrintUsage(const std::string& header, const reflection::ReflectableClass& object);
  static void PrintDefaultUsage(const std::string& programName, const reflection::ReflectableClass& object);
};

}

#endif /* ARGUMENTSPARSER_H_ */
