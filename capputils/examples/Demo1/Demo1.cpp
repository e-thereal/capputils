/**
 * \brief Shows how to read parameters from the command line.
 * 
 * \example Demo1/Demo1.cpp
 * 
 * This is a first example.
 */


/*** Begin: DataModel.h ***/

#include <capputils/ReflectableClass.h>
#include <capputils/ArgumentsParser.h>

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

int main(int argc, char* argv[])
{
  using namespace capputils;

  DataModel model;

  ArgumentsParser::Parse(model, argc, argv);
  if (model.getHelp()) {
    ArgumentsParser::PrintDefaultUsage("Demo1", model);
    return 0;
  }

  std::cout << "InputName: " << model.getInputName() << std::endl;
  std::cout << "OuputName: " << model.getOutputName() << std::endl;
  std::cout << "ParameterName: " << model.getParameter() << std::endl;

	return 0;
}
