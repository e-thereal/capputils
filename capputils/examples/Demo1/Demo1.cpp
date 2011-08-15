/**
 * \brief Shows how to read parameters from the command line.
 * 
 * \example Demo1/Demo1.cpp
 *
 * This example demonstrates the basics of argument parsing using \c capputils. The program has 4 parameters:
 * - An input name of type \c string
 * - An output name of type \c string
 * - A parameter of type \c int
 * - A parameter help of type \c bool.
 *
 * The help parameter is somewhat special. It is used as a command line flag that is either active or not.
 * If it is active, its value is set to \c true and \c false otherwise. The property has the attribute
 * \c Flag() that indicates that this parameter is not followed by a value while paring for command line
 * arguments. The example also features the \c DescriptionAttribute. This attribute can be used to
 * add a description to a property within code. It is used to print usage information.
 *
 * The program generates the following outputs:
 * \verbatim
> Demo1.exe --Help
 
Usage: Demo1 [switches], where switches are:

  --InputName    Name of the input file.
  --OutputName   Name of the output file.
  --Parameter    Parameter of type int.
  --Help         Shows this help.
\endverbatim
 * \verbatim
> Demo1.exe --InputName Hello --OutputName World --Parameter 42

InputName: Hello
OuputName: World
Parameter: 42
\endverbatim
 */


/*** Begin: DataModel.h ***/

#include <capputils/ReflectableClass.h>

class DataModel : public capputils::reflection::ReflectableClass {
  InitReflectableClass(DataModel)

  Property(InputName, std::string)
  Property(OutputName, std::string)
  Property(Parameter, int)
  Property(Help, bool)

public:
  DataModel();
};

/*** End: DataModel.h ***/

/*** Begin: DataModel.cpp ***/

#include <capputils/DescriptionAttribute.h>
#include <capputils/FlagAttribute.h>

BeginPropertyDefinitions(DataModel)
  using namespace capputils::attributes;

  DefineProperty(InputName, Description("Name of the input file."))
  DefineProperty(OutputName, Description("Name of the output file."))
  DefineProperty(Parameter, Description("Parameter of type int."))
  DefineProperty(Help, Description("Shows this help."), Flag())
  
EndPropertyDefinitions

DataModel::DataModel() : _InputName(""), _OutputName(""), _Parameter(0), _Help(false) { }

/*** End: DataModel.cpp ***/

#include <iostream>

#include <capputils/ArgumentsParser.h>

int main(int argc, char* argv[])
{
  using namespace capputils;

  DataModel model;

  ArgumentsParser::Parse(model, argc, argv);
  if (model.getHelp()) {
    ArgumentsParser::PrintDefaultUsage("Demo1", model);
    return 0;
  }

  std::cout << std::endl;
  std::cout << "InputName: " << model.getInputName() << std::endl;
  std::cout << "OuputName: " << model.getOutputName() << std::endl;
  std::cout << "Parameter: " << model.getParameter() << std::endl;

	return 0;
}
