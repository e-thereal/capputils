/*
 * Xmlizer.h
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_XMLIZER_H_
#define CAPPUTILS_XMLIZER_H_

#include <tinyxml/tinyxml.h>
#include <iostream>
#include <string>

#include <capputils/reflection/ReflectableClass.h>

namespace capputils {

std::string makeXmlName(const std::string& className);
std::string makeClassName(const std::string& xmlName);

class Xmlizer {
public:
  static void AddPropertyToXml(TiXmlNode& xmlNode, const reflection::ReflectableClass& object,
      const reflection::IClassProperty* property);
  static TiXmlElement* CreateXml(const reflection::ReflectableClass& object);
  static void ToXml(TiXmlNode& xmlNode, const reflection::ReflectableClass& object);
  static void ToXml(const ::std::string& filename, const reflection::ReflectableClass& object);
  static void ToXml(std::ostream& os, const reflection::ReflectableClass& object);
  static void ToFile(const ::std::string& filename, TiXmlNode* node);
  static void ToDocument(std::ostream& os, TiXmlNode* node);

  static void GetPropertyFromXml(reflection::ReflectableClass& object,
      reflection::IClassProperty* property, TiXmlNode& node);
  static void GetPropertyFromXml(reflection::ReflectableClass& object,
      reflection::IClassProperty* property, const std::string& filename);

  static reflection::ReflectableClass* CreateReflectableClass(TiXmlNode& xmlNode);
  static reflection::ReflectableClass* CreateReflectableClass(const ::std::string& filename);
  static void FromXml(reflection::ReflectableClass& object, TiXmlNode& xmlNode);
  static void FromXml(reflection::ReflectableClass& object, const ::std::string& filename);
  static void FromXmlString(reflection::ReflectableClass& object, const std::string& xmlString);
};

}

#endif /* XMLIZER_H_ */
