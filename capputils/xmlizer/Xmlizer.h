/*
 * Xmlizer.h
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#ifndef XMLIZER_H_
#define XMLIZER_H_

#include <tinyxml.h>
#include <iostream>
#include <string>

#include "ReflectableClass.h"

namespace xmlizer {

class Xmlizer {
public:
  static TiXmlElement* CreateXml(const ::reflection::ReflectableClass& object);
  static void ToXml(TiXmlNode& xmlNode, const ::reflection::ReflectableClass& object);
  static void ToXml(const ::std::string& filename, const ::reflection::ReflectableClass& object);

  static void FromXml(::reflection::ReflectableClass& object, const TiXmlNode& xmlNode);
  static void FromXml(::reflection::ReflectableClass& object, const ::std::string& filename);
};

}

#endif /* XMLIZER_H_ */
