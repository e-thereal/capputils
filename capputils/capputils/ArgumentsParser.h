/**
 * \brief Contains the parser module for command line arguments.
 * \file ArgumentsParser.h
 *
 *  \date Feb 10, 2011
 *  \author Tom Brosch
 */

#ifndef CAPPUTILS_ARGUMENTSPARSER_H_
#define CAPPUTILS_ARGUMENTSPARSER_H_

#include "capputils.h"
#include "ReflectableClass.h"

namespace capputils {

/** \brief This class contains static methods in order to parse command line arguments and to show a usage message
 *
 *  Command line arguments have the form \c --PropertyName PropertyValue or just \c --FlagName if the
 *  property \c FlagName has the \c FlagAttribute
 */
class CAPPUTILS_API ArgumentsParser {
public:
  /**
   * \brief Parses command line arguments and sets the properties of a class accordingly
   *
   * \param[out]  object  Reference to an instance of a \c ReflectableClass. Command line arguments
   *                      are matched with the properties of the instance \a object.
   * \param[in]   argc    Number of command line arguments. The first argument is the program name itself
   * \param[in]   argv    Array of \c char* containing the command line arguments
   */
  static void Parse(reflection::ReflectableClass& object, int argc, char** argv, bool parseOnlyParameter = false);

  /**
   * \brief Prints a usage message to standard output using a user specified header.
   *
   * \param[in] header  First list of the usage method.
   * \param[in] object  All properties of that object are printed with according descriptions
   *
   * The \c DescriptionAttribute of a property is used to provide a meaningful description of
   * the according parameter.
   */
  static void PrintUsage(const std::string& header, const reflection::ReflectableClass& object, bool showOnlyParameters = false);

  /**
   * \brief Prints a usage message to standard output with a default header.
   *
   * \param[in] programName The name of the program. Used as part of the first line of the usage message.
   * \param[in] object      All properties of that object are printed with according descriptions.
   *
   * The \c DescriptionAttribute of a property is used to provide a meaningful description of
   * the according parameter.
   */
  static void PrintDefaultUsage(const std::string& programName, const reflection::ReflectableClass& object, bool showOnlyParameters = false);
};

}

#endif /* CAPPUTILS_ARGUMENTSPARSER_H_ */
