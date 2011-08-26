#pragma once
#ifndef _CAPPUTILS_CAPPUTILS_H_
#define _CAPPUTILS_CAPPUTILS_H_

#ifdef _WIN32
#ifdef CAPPUTILS_EXPORTS
//#define CAPPUTILS_API __declspec(dllexport)
#else
#pragma comment(lib, "capputils")
//#define CAPPUTILS_API __declspec(dllimport)
#endif
#else
//#define CAPPUTILS_API
#endif

#define CAPPUTILS_API

#ifndef CAPPUTILS_COMMA
#define CAPPUTILS_COMMA() ,
#endif

/**
 * \mainpage \c capputils - A \c C++ toolkit for console applications
 *
 * \section sec_intro Introduction
 *
 * Handles program parameters, arguments, configuration files. Adds simple reflection.
 *
 * \subsection sec_prerequisites Prerequisites
 *
 * You need the following additional libraries:
 *
 * - Boost: http://www.boost.org/
 * - Tinyxml: http://www.grinninglizard.com/tinyxml/ (The library is way better than how the website looks at the first gaze) 
 *
 * \subsection sec_website Project Website
 *
 * - http://code.google.com/p/capputils/
 *
 * \subsection sec_related Related Projects
 *
 * - Gapputils: http://code.google.com/p/gapputils 
 * 
 * \section sec_concept Concept
 *
 * \subsection sec_intro Introduction
 *
 * The parameter handling concept of \c capputils assumes that parameters are described
 * by a class. Instances of that class serve as the parameter storage. If not declared
 * otherwise, every property of that class is treaten as a parameter whose value can
 * be set and read through various parameter handling mechanisms. These include but are not
 * limited to:
 * - Get the value of a parameter using its getter method
 * - Get the value of a parameter as a \c string using its name
 * - Write the value of a parameter to a configuration file
 * - Set a parameter using its setter method
 * - Set a parameter to a value, obtained as a program argument
 * - Set a parameter according to a configuration file
 * - Set a parameter using its name
 * 
 * In order to facilitate automatic parameter handling, \c capputils must be aware of
 * the declaration of the class at runtime. This awareness is achieved through
 * reflection. Reflection allows for the following features at runtime
 * - Query all defined properties of an instance of a class
 * - Get the type of a property
 * - Get attributes of a property
 * - Set the value of a property knowing its name or ID
 * - Get the value of a property knowing its name or ID
 *
 * \subsection sec_reflection Reflection
 *
 * Since \c C++ does not support reflection per se, a class has to be made reflectable.
 * Adding reflection for a class is done using the following steps:
 * - The class must be directly or indirectly inherited from \c ReflectableClass
 * - You must call the macro \c InitReflectableClass() with the class name at the
 *   beginning of the declaration of the class
 * - All properties are declared using the \c Property macro
 * - All properties must be defined in the source (CPP) file
 * - The macro \c BeginPropertyDefinitions() marks the beginning of all property definitions of one class and must
 *   must be called once for every reflectable class. (even if that class does not define any properties)
 * - The macro \c EndPropertyDefinitions marks the end of all property definitions of one class
 * - Properties are defined using the \c DefineProperty macro. You can declare a property without defining it.
 *   Declaring a property through \c Property generates appropriate getter and setter methods but the
 *   property will not be accessable through reflection unless you define it using \c DefineProperty.
 * - Use the macro \c ReflectableProperty instead of \c DefineProperty if the property type is derived from
 *   \c ReflectableClass.
 *
 * The first demo \c Demo1 in the examples section provides a simple example that should be used as a
 * starting point.
 */

#endif
