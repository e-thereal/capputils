/**
 * \brief Contains the parser module for command line arguments.
 * \file ArgumentsParser.h
 *
 *  \date Feb 10, 2011
 *  \author Tom Brosch
 */

#ifndef CAPPUTILS_ARGUMENTSPARSER_H_
#define CAPPUTILS_ARGUMENTSPARSER_H_

#include <capputils/capputils.h>
#include <capputils/reflection/ReflectableClass.h>

namespace capputils {

struct ParameterDescription {
  reflection::ReflectableClass* object;
  reflection::IClassProperty* property;
  std::string longName, shortName, description;

  ParameterDescription();

  ParameterDescription(reflection::ReflectableClass* object, reflection::IClassProperty* property,
      const std::string& longName, const std::string& shortName, const std::string& description);
};

struct ParameterDescriptions {
  std::vector<ParameterDescription> parameters;
  boost::shared_ptr<ParameterDescription> operands;
};

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
  static boost::shared_ptr<std::vector<std::string> > Parse(reflection::ReflectableClass& object, int argc, char** argv, bool parseOnlyParameter = false);

  /**
   * \brief Parses command line arguments and sets the properties of a class accordingly
   *
   * \param[out]  parameters  List of parameters, for which a summary should be printed.
   * \param[in]   argc        Number of command line arguments. The first argument is the program name itself
   * \param[in]   argv        Array of \c char* containing the command line arguments
   */
  static boost::shared_ptr<std::vector<std::string> > Parse(ParameterDescriptions& parameters, int argc, char** argv, bool parseOnlyParameter = false);

  /**
   * \brief Prints a usage message to standard output using a user specified header.
   *
   * \param[in] header  First line of the usage method.
   * \param[in] object  All properties of that object are printed with according descriptions
   *
   * The \c DescriptionAttribute of a property is used to provide a meaningful description of
   * the according parameter.
   */
  static void PrintUsage(const std::string& header, reflection::ReflectableClass& object, bool showOnlyParameters = false);

  /**
   * \brief Prints a usage message to standard output using a user specified header.
   *
   * \param[in] header      First line of the usage method.
   * \param[in] parameters  List of parameters, for which a summary should be printed.
   */
  static void PrintUsage(const std::string& header, ParameterDescriptions& parameters);

  /**
   * \brief Prints a usage message to standard output with a default header.
   *
   * \param[in] programName The name of the program. Used as part of the first line of the usage message.
   * \param[in] object      All properties of that object are printed with according descriptions.
   *
   * The \c DescriptionAttribute of a property is used to provide a meaningful description of
   * the according parameter.
   */
  static void PrintDefaultUsage(const std::string& programName, reflection::ReflectableClass& object, bool showOnlyParameters = false);

  /**
   * \brief Prints a usage message to standard output with a default header.
   *
   * \param[in] programName The name of the program. Used as part of the first line of the usage message.
   * \param[in] parameters  List of parameters, for which a summary should be printed.
   */
  static void PrintDefaultUsage(const std::string& programName, ParameterDescriptions& parameters);

  static void CreateParameterDescriptions(reflection::ReflectableClass& object, bool includeOnlyParameters, ParameterDescriptions& parameters);
};

}

#endif /* CAPPUTILS_ARGUMENTSPARSER_H_ */
