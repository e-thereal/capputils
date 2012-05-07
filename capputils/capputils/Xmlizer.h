/*
 * Xmlizer.h
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#ifndef XMLIZER_H_
#define XMLIZER_H_

#include <tinyxml/tinyxml.h>
#include <iostream>
#include <string>

#include "ReflectableClass.h"

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

  static void GetPropertyFromXml(reflection::ReflectableClass& object,
      reflection::IClassProperty* property, const TiXmlNode& node);
  static void GetPropertyFromXml(reflection::ReflectableClass& object,
      reflection::IClassProperty* property, const std::string& filename);
  static reflection::ReflectableClass* CreateReflectableClass(const TiXmlNode& xmlNode);
  static reflection::ReflectableClass* CreateReflectableClass(const ::std::string& filename);
  static void FromXml(reflection::ReflectableClass& object, const TiXmlNode& xmlNode);
  static void FromXml(reflection::ReflectableClass& object, const ::std::string& filename);
};

}

#endif /* XMLIZER_H_ */
